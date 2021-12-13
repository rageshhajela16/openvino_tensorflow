/*****************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
*****************************************************************************/

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <ie_core.hpp>
#include "ngraph/ngraph.hpp"
#include <tensorflow_frontend/graph_iterator.hpp>
#include <tensorflow_frontend/decoder.hpp>

#include "tensorflow/core/graph/graph.h"

using namespace std;

namespace tensorflow {
namespace openvino_tensorflow {


struct ovtf_attr {
  int type;
};

// A Inference Engine executable object produced by compiling an nGraph
// function.
class OVTFDecoder : public ov::frontend::DecoderBase {
public:
    OVTFDecoder(string op_type, string op_name, int input_size) : m_op_type(op_type), m_op_name(op_name), m_input_size(input_size), m_next_ptr(nullptr) {}

    //std::shared_ptr<ov::Variant> get_attribute(const std::string& name,
    //                                           const ngraph::VariantTypeInfo& type_info) const override;
    ov::Any get_attribute(const std::string& name, const std::type_info& type_info) const override;

    size_t get_input_size() const override;

    void get_input_node(const size_t input_port_idx,
                        std::string& producer_name,
                        size_t& producer_output_port_index) const override;

    const std::string& get_op_type() const override;

    const std::string& get_op_name() const override;

    void set_next(shared_ptr<OVTFDecoder> next_ptr);
    shared_ptr<OVTFDecoder> get_next() const;

    void add_attr(std::string attr_name, ovtf_attr value);

    void set_tfnode(shared_ptr<tensorflow::Node> tfnode_ptr);
    vector<::tensorflow::AttrValue> decode_attribute_helper(const string& name) const;

private:
    string m_op_name;
    string m_op_type;
    int m_input_size;
    shared_ptr<OVTFDecoder> m_next_ptr;
    std::map<std::string, ovtf_attr> m_attr_map;
    shared_ptr<tensorflow::Node> m_tfnode_ptr;
};
}  // namespace openvino_tensorflow
}  // namespace tensorflow
