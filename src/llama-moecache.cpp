#include "llama-moecache.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>

namespace {

struct layer_state {
    llama_moe_cache_layer pub;

    std::vector<int32_t>  slot_expert; // slot -> expert id, -1 when empty
    std::vector<int32_t>  expert_slot; // expert id -> slot, -1 when uncached
    std::vector<uint64_t> request_use; // expert id -> uses during this generation

    uint64_t n_hit  = 0;
    uint64_t n_miss = 0;
};

struct moe_cache {
    int32_t n_slots = 0;
    int32_t n_active_generations = 0;

    std::mutex mtx;

    std::vector<layer_state> layers;
    std::map<const ggml_tensor *, size_t> by_up_src;

    std::vector<ggml_context *>         ctxs;
    std::vector<ggml_backend_buffer_t>  bufs;

    uint64_t n_cpu_calls     = 0;
    uint64_t n_cpu_selected  = 0;
    uint64_t n_cpu_skipped   = 0;
    uint64_t n_cpu_vec_dot   = 0;
    uint64_t n_cpu_converted = 0;
    uint64_t n_cpu_fast_path_calls = 0;
    uint64_t n_cpu_fast_path_us    = 0;
};

moe_cache * g_cache = nullptr;
std::mutex g_init_mtx;
bool g_init_done = false;

int parse_layer_from_name(const char * name) {
    // "blk.<il>.ffn_gate_exps.weight"
    if (strncmp(name, "blk.", 4) != 0) {
        return -1;
    }
    return atoi(name + 4);
}

void moe_obs_cb(const char * name, const struct ggml_tensor * ids, void * ud) {
    moe_cache * mc = (moe_cache *) ud;

    const int64_t n_ids    = ids->ne[0];
    const int64_t n_tokens = ids->ne[1];
    if (n_tokens != 1) {
        return;
    }

    const int il = parse_layer_from_name(name);
    if (il < 0) {
        return;
    }

    layer_state * ls = nullptr;
    for (auto & l : mc->layers) {
        if (l.pub.il == il) { ls = &l; break; }
    }
    if (!ls) {
        return;
    }

    std::lock_guard<std::mutex> lock(mc->mtx);
    if (mc->n_active_generations == 0) {
        return;
    }
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t i = 0; i < n_ids; ++i) {
            const int32_t id = *(const int32_t *) ((const char *) ids->data + t*ids->nb[1] + i*ids->nb[0]);
            if (id < 0 || id >= (int32_t) ls->expert_slot.size()) {
                continue;
            }
            ls->request_use[id]++;
            const int32_t slot = ls->expert_slot[id];
            if (slot >= 0) {
                ls->n_hit++;
            } else {
                ls->n_miss++;
            }
        }
    }
}

void moe_cpu_stats_cb(const char * name, uint64_t n_selected, uint64_t n_skipped, uint64_t n_vec_dot, uint64_t n_converted, uint64_t fast_path_us, void * ud) {
    GGML_UNUSED(name);
    moe_cache * mc = (moe_cache *) ud;

    std::lock_guard<std::mutex> lock(mc->mtx);
    mc->n_cpu_calls++;
    mc->n_cpu_selected  += n_selected;
    mc->n_cpu_skipped   += n_skipped;
    mc->n_cpu_vec_dot   += n_vec_dot;
    mc->n_cpu_converted += n_converted;
    if (n_selected == n_skipped) {
        mc->n_cpu_fast_path_calls++;
        mc->n_cpu_fast_path_us += fast_path_us;
    }
}

void upload_slice(ggml_tensor * dst_c, const ggml_tensor * src, int32_t expert, int32_t slot) {
    const size_t sz = src->nb[2];
    if ((size_t) slot*dst_c->nb[2] + sz > ggml_nbytes(dst_c) || (size_t) expert*sz + sz > ggml_nbytes(src)) {
        LLAMA_LOG_ERROR("moe-cache: bad upload %s <- %s expert=%d slot=%d sz=%zu dst_nb2=%zu dst_bytes=%zu src_bytes=%zu\n",
                dst_c->name, src->name, expert, slot, sz, dst_c->nb[2], ggml_nbytes(dst_c), ggml_nbytes(src));
        return;
    }
    ggml_backend_tensor_set(dst_c, (const char *) src->data + (size_t) expert*sz, (size_t) slot*dst_c->nb[2], sz);
}

void set_table_entry(llama_moe_cache_layer & pub, int32_t expert, int32_t slot_or_dummy) {
    const int32_t v = slot_or_dummy;
    ggml_backend_tensor_set(pub.dev_table,  &v, (size_t) expert*sizeof(int32_t), sizeof(int32_t));
    ggml_backend_tensor_set(pub.host_table, &v, (size_t) expert*sizeof(int32_t), sizeof(int32_t));
}

} // namespace

void llama_moe_cache_init(const llama_model & model, int32_t n_slots) {
    std::lock_guard<std::mutex> init_lock(g_init_mtx);
    if (g_init_done) {
        return;
    }
    [&]() {
        if (n_slots <= 0) {
            g_init_done = true;
            return;
        }

        auto * mc = new moe_cache();
        mc->n_slots = n_slots;

        // collect the host-resident expert layers, grouped by the device buffer
        // type of that layer's router (the cache lives next to the router)
        struct cand { int il; const llama_layer * l; };
        std::map<ggml_backend_buffer_type_t, std::vector<cand>> groups;

        for (size_t il = 0; il < model.layers.size(); ++il) {
            const auto & l = model.layers[il];
            if (!l.ffn_up_exps || !l.ffn_gate_exps || !l.ffn_down_exps || !l.ffn_gate_inp) {
                continue;
            }
            if (!l.ffn_up_exps->data || !l.ffn_gate_exps->data || !l.ffn_down_exps->data) {
                continue; // dry-run / memory-estimation model: weights not loaded, don't bind to it
            }
            if (!l.ffn_up_exps->buffer || !ggml_backend_buffer_is_host(l.ffn_up_exps->buffer)) {
                continue; // experts already on a device: nothing to cache
            }
            if (!l.ffn_gate_inp->buffer || ggml_backend_buffer_is_host(l.ffn_gate_inp->buffer)) {
                continue; // no device home for the cache
            }
            groups[ggml_backend_buffer_get_type(l.ffn_gate_inp->buffer)].push_back({(int) il, &l});
        }

        if (groups.empty()) {
            LLAMA_LOG_INFO("%s: LLAMA_MOE_CACHE_SLOTS=%d but no host-resident expert layers found - disabled\n", __func__, n_slots);
            delete mc;
            return;
        }

        // host buffer for the CPU-side tables
        std::vector<cand> all;
        for (auto & g : groups) {
            all.insert(all.end(), g.second.begin(), g.second.end());
        }

        auto alloc_group = [&](ggml_backend_buffer_type_t buft, const std::vector<cand> & cands, bool tables_only) -> bool {
            ggml_init_params ip = {
                /*.mem_size  =*/ ggml_tensor_overhead()*(cands.size()*4 + 8),
                /*.mem_buffer=*/ nullptr,
                /*.no_alloc  =*/ true,
            };
            ggml_context * ctx = ggml_init(ip);
            if (!ctx) {
                return false;
            }
            mc->ctxs.push_back(ctx);

            for (const auto & c : cands) {
                layer_state * ls = nullptr;
                for (auto & l : mc->layers) {
                    if (l.pub.il == c.il) { ls = &l; break; }
                }
                if (!ls) {
                    mc->layers.push_back({});
                    ls = &mc->layers.back();
                    ls->pub.il       = c.il;
                    ls->pub.n_slots  = n_slots;
                    ls->pub.up_src   = c.l->ffn_up_exps;
                    ls->pub.gate_src = c.l->ffn_gate_exps;
                    ls->pub.down_src = c.l->ffn_down_exps;
                }

                if (tables_only) {
                    ls->pub.host_table = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, ls->pub.up_src->ne[2]);
                    ggml_format_name(ls->pub.host_table, "moe_cache_htbl.%d", c.il);
                } else {
                    const ggml_tensor * u = c.l->ffn_up_exps;
                    const ggml_tensor * g = c.l->ffn_gate_exps;
                    const ggml_tensor * d = c.l->ffn_down_exps;
                    ls->pub.up_c   = ggml_new_tensor_3d(ctx, u->type, u->ne[0], u->ne[1], n_slots + 1);
                    ls->pub.gate_c = ggml_new_tensor_3d(ctx, g->type, g->ne[0], g->ne[1], n_slots + 1);
                    ls->pub.down_c = ggml_new_tensor_3d(ctx, d->type, d->ne[0], d->ne[1], n_slots + 1);
                    ls->pub.dev_table = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, u->ne[2]);
                    // Use source names so tensor split applies the same sharding rules.
                    ggml_set_name(ls->pub.up_c,   u->name);
                    ggml_set_name(ls->pub.gate_c, g->name);
                    ggml_set_name(ls->pub.down_c, d->name);
                    ggml_format_name(ls->pub.dev_table, "moe_cache_tbl.%d",  c.il);
                }
            }

            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
            if (!buf) {
                LLAMA_LOG_WARN("%s: failed to allocate MoE cache buffer on %s - cache disabled\n",
                        __func__, ggml_backend_buft_name(buft));
                return false;
            }
            ggml_backend_buffer_clear(buf, 0);
            mc->bufs.push_back(buf);
            return true;
        };

        bool ok = alloc_group(ggml_backend_cpu_buffer_type(), all, /*tables_only=*/true);
        for (auto & g : groups) {
            if (!ok) {
                break;
            }
            ok = alloc_group(g.first, g.second, /*tables_only=*/false);
        }

        if (!ok) {
            for (auto * b : mc->bufs) { ggml_backend_buffer_free(b); }
            for (auto * c : mc->ctxs) { ggml_free(c); }
            delete mc;
            g_init_done = true; // a real model was seen and allocation failed: stay disabled
            return;
        }

        // init state + tables (everything uncached -> dummy slot n_slots)
        size_t vram = 0;
        for (auto & ls : mc->layers) {
            const int64_t n_expert = ls.pub.up_src->ne[2];
            ls.slot_expert.assign(n_slots, -1);
            ls.expert_slot.assign(n_expert, -1);
            ls.request_use.assign(n_expert, 0);

            std::vector<int32_t> dummy(n_expert, n_slots);
            ggml_backend_tensor_set(ls.pub.dev_table,  dummy.data(), 0, n_expert*sizeof(int32_t));
            ggml_backend_tensor_set(ls.pub.host_table, dummy.data(), 0, n_expert*sizeof(int32_t));

            mc->by_up_src[ls.pub.up_src] = &ls - mc->layers.data();
            vram += ggml_nbytes(ls.pub.up_c) + ggml_nbytes(ls.pub.gate_c) + ggml_nbytes(ls.pub.down_c);
            LLAMA_LOG_DEBUG("moe-cache: init layer %d '%s' %zu bytes/expert\n",
                    ls.pub.il, ls.pub.up_src->name, ls.pub.up_src->nb[2]);
        }

        ggml_set_moe_obs_callback(moe_obs_cb, mc);
        ggml_set_moe_cpu_stats_callback(moe_cpu_stats_cb, mc);
        g_cache = mc;
        g_init_done = true;

        LLAMA_LOG_INFO("%s: MoE expert cache enabled: %zu layers x %d slots, %.1f MiB device memory\n",
                __func__, mc->layers.size(), n_slots, vram/1024.0/1024.0);
    }();
}

const llama_moe_cache_layer * llama_moe_cache_lookup(const ggml_tensor * up_exps) {
    if (!g_cache) {
        return nullptr;
    }
    auto it = g_cache->by_up_src.find(up_exps);
    if (it == g_cache->by_up_src.end()) {
        return nullptr;
    }
    return &g_cache->layers[it->second].pub;
}

llama_moe_cache_stats llama_moe_cache_get_stats() {
    llama_moe_cache_stats stats = {};
    moe_cache * mc = g_cache;
    if (!mc) {
        return stats;
    }

    std::lock_guard<std::mutex> lock(mc->mtx);
    stats.n_layers = (int32_t) mc->layers.size();
    stats.n_slots  = mc->n_slots;
    stats.n_cpu_calls     = mc->n_cpu_calls;
    stats.n_cpu_selected  = mc->n_cpu_selected;
    stats.n_cpu_skipped   = mc->n_cpu_skipped;
    stats.n_cpu_vec_dot   = mc->n_cpu_vec_dot;
    stats.n_cpu_converted = mc->n_cpu_converted;
    stats.n_cpu_fast_path_calls = mc->n_cpu_fast_path_calls;
    stats.n_cpu_fast_path_us    = mc->n_cpu_fast_path_us;
    for (const auto & ls : mc->layers) {
        stats.n_hit  += ls.n_hit;
        stats.n_miss += ls.n_miss;
    }
    return stats;
}

void llama_moe_cache_begin_generation() {
    moe_cache * mc = g_cache;
    if (!mc) {
        return;
    }

    std::lock_guard<std::mutex> lock(mc->mtx);
    if (mc->n_active_generations == 0) {
        for (auto & ls : mc->layers) {
            std::fill(ls.request_use.begin(), ls.request_use.end(), 0);
        }
    }
    mc->n_active_generations++;
}

void llama_moe_cache_end_generation() {
    moe_cache * mc = g_cache;
    if (!mc) {
        return;
    }

    std::lock_guard<std::mutex> lock(mc->mtx);
    if (mc->n_active_generations == 0) {
        return;
    }
    mc->n_active_generations--;
    if (mc->n_active_generations > 0) {
        return;
    }

    for (auto & ls : mc->layers) {
        std::vector<int32_t> incoming;
        incoming.reserve(ls.request_use.size());
        for (int32_t expert = 0; expert < (int32_t) ls.request_use.size(); ++expert) {
            if (ls.expert_slot[expert] < 0 && ls.request_use[expert] > 0) {
                incoming.push_back(expert);
            }
        }
        std::sort(incoming.begin(), incoming.end(), [&](int32_t a, int32_t b) {
            if (ls.request_use[a] != ls.request_use[b]) {
                return ls.request_use[a] > ls.request_use[b];
            }
            return a < b;
        });

        std::vector<int32_t> victims(mc->n_slots);
        for (int32_t slot = 0; slot < mc->n_slots; ++slot) {
            victims[slot] = slot;
        }
        std::sort(victims.begin(), victims.end(), [&](int32_t a, int32_t b) {
            const int32_t expert_a = ls.slot_expert[a];
            const int32_t expert_b = ls.slot_expert[b];
            if (expert_a < 0 || expert_b < 0) {
                if (expert_a < 0 && expert_b < 0) {
                    return a < b;
                }
                return expert_a < 0;
            }
            if (ls.request_use[expert_a] != ls.request_use[expert_b]) {
                return ls.request_use[expert_a] < ls.request_use[expert_b];
            }
            return a < b;
        });

        const size_t n_replace = std::min(incoming.size(), victims.size());
        for (size_t i = 0; i < n_replace; ++i) {
            const int32_t expert = incoming[i];
            const int32_t slot   = victims[i];
            const int32_t victim = ls.slot_expert[slot];
            if (victim >= 0 && ls.request_use[expert] <= ls.request_use[victim]) {
                break;
            }
            if (victim >= 0) {
                ls.expert_slot[victim] = -1;
                ls.slot_expert[slot]   = -1;
                set_table_entry(ls.pub, victim, mc->n_slots);
            }

            upload_slice(ls.pub.up_c,   ls.pub.up_src,   expert, slot);
            upload_slice(ls.pub.gate_c, ls.pub.gate_src, expert, slot);
            upload_slice(ls.pub.down_c, ls.pub.down_src, expert, slot);

            ls.slot_expert[slot]   = expert;
            ls.expert_slot[expert] = slot;
            set_table_entry(ls.pub, expert, slot);
        }
        std::fill(ls.request_use.begin(), ls.request_use.end(), 0);
    }
}
