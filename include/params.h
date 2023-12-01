#pragma once

#include <unordered_map>
#include <functional>

#include "tensor.h"

namespace tllm {
namespace detail {

class ParamsDict : public std::unordered_map<std::string, std::reference_wrapper<Tensor>> {
public:
    ParamsDict() = default;
    ParamsDict(std::initializer_list<value_type> items)
        : std::unordered_map<std::string, std::reference_wrapper<Tensor>>(items) { }

    ParamsDict(std::initializer_list<std::pair<std::string, ParamsDict>> dicts) {
        for(auto& named_dict: dicts) {
            auto& name = named_dict.first;
            auto& dict = named_dict.second;

            for(auto iter: dict)
                this->insert({
                    name + "." + iter.first,
                    iter.second
                });
        }
    }

    void insert_parmdict(string name, ParamsDict parms) {
        for(auto iter: parms)
            this->insert({
                name + "." + iter.first,
                iter.second
            });
    }

    Tensor& operator[](const std::string& key) {
        auto iter = find(key);
        return iter->second.get();
    }
};

}
};