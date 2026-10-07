#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

using l0xre_distribution = std::vector<std::pair<int32_t, double>>;
inline l0xre_distribution l0xre_normalize(l0xre_distribution d) {
    std::sort(d.begin(), d.end());
    l0xre_distribution out;
    for (const auto & x : d) {
        if (!std::isfinite(x.second) || x.second < 0) throw std::runtime_error("invalid probability");
        if (x.second == 0) continue;
        if (!out.empty() && out.back().first == x.first) out.back().second += x.second;
        else out.push_back(x);
    }
    long double sum=0;for(const auto & x:out)sum+=x.second;
    if (!(sum>0)) throw std::runtime_error("empty probability distribution");
    for(auto & x:out)x.second=double(x.second/sum);
    return out;
}
inline double l0xre_probability(const l0xre_distribution & d,int32_t token) {
    auto it=std::lower_bound(d.begin(),d.end(),token,[](const auto & x,int32_t id){return x.first<id;});
    return it!=d.end() && it->first==token ? it->second : 0;
}
inline int32_t l0xre_draw(const l0xre_distribution & d,std::mt19937_64 & rng) {
    const double u=std::generate_canonical<double,53>(rng);
    double cumulative=0;
    for(const auto & x:d){cumulative+=x.second;if(u<cumulative)return x.first;}
    if(d.empty())throw std::runtime_error("cannot sample empty distribution");
    return d.back().first;
}
struct l0xre_rejection_result {int32_t token;bool accepted;};
inline l0xre_rejection_result l0xre_reject(
        const l0xre_distribution & p,const l0xre_distribution & q,
        int32_t proposal,std::mt19937_64 & rng) {
    const double qp=l0xre_probability(q,proposal),pp=l0xre_probability(p,proposal);
    if(!(qp>0))throw std::runtime_error("proposal outside recorded draft support");
    const double u=std::generate_canonical<double,53>(rng);
    if(u*qp<pp)return {proposal,true};
    l0xre_distribution residual;residual.reserve(p.size());
    for(const auto & x:p)residual.emplace_back(x.first,std::max(0.0,x.second-l0xre_probability(q,x.first)));
    return {l0xre_draw(l0xre_normalize(std::move(residual)),rng),false};
}
