
#include <openvino_tensorflow/ovtf_decoder.h>
#include <ngraph/ngraph.hpp>
#include <openvino/core/type/element_type.hpp>
#include <string>
#include <tensorflow_frontend/decoder.hpp>
#include <tensorflow_frontend/frontend.hpp>
#include <vector>



namespace tensorflow {
namespace openvino_tensorflow {


#ifndef TF_FE_NO_TF_DEP
namespace {
const std::map<::tensorflow::DataType, ov::element::Type>& TYPE_MAP() {
    static const std::map<::tensorflow::DataType, ov::element::Type> type_map{
        {::tensorflow::DataType::DT_BOOL, ov::element::boolean},
        {::tensorflow::DataType::DT_INT16, ov::element::i16},
        {::tensorflow::DataType::DT_INT32, ov::element::i32},
        {::tensorflow::DataType::DT_INT64, ov::element::i64},
        {::tensorflow::DataType::DT_HALF, ov::element::f16},
        {::tensorflow::DataType::DT_FLOAT, ov::element::f32},
        {::tensorflow::DataType::DT_DOUBLE, ov::element::f64},
        {::tensorflow::DataType::DT_UINT8, ov::element::u8},
        {::tensorflow::DataType::DT_INT8, ov::element::i8},
        {::tensorflow::DataType::DT_BFLOAT16, ov::element::bf16}};
    return type_map;
}
}  // namespace
#endif


ov::Any OVTFDecoder::get_attribute(const std::string& name, const std::type_info& type_info) const {
#ifdef TF_FE_NO_TF_DEP
    return {};
#else
    auto attrs = decode_attribute_helper(name);
    if (attrs.empty()) {
        return {};
    }

    if (type_info == typeid(std::string)) {
        return attrs[0].s();
    } else if (type_info == typeid(int64_t)) {
        return attrs[0].i();
    } else if (type_info == typeid(std::vector<int64_t>)) {
        std::vector<int64_t> longs;
        longs.reserve(attrs[0].list().i_size());
        for (size_t idx = 0; idx < attrs[0].list().i_size(); ++idx) {
            longs.push_back(attrs[0].list().i(idx));
        }
        return longs;
    } else if (type_info == typeid(int32_t)) {
        return static_cast<int32_t>(attrs[0].i());
    } else if (type_info == typeid(std::vector<int32_t>)) {
        std::vector<int32_t> ints;
        ints.reserve(attrs[0].list().i_size());
        for (size_t idx = 0; idx < attrs[0].list().i_size(); ++idx) {
            ints.push_back(static_cast<int32_t>(attrs[0].list().i(idx)));
        }
        return ints;
    } else if (type_info == typeid(float)) {
        return attrs[0].f();
    } else if (type_info == typeid(std::vector<float>)) {
        std::vector<float> floats;
        floats.reserve(attrs[0].list().i_size());
        for (size_t idx = 0; idx < attrs[0].list().i_size(); ++idx) {
            floats.push_back(attrs[0].list().f(idx));
        }
        return floats;
    } else if (type_info == typeid(ov::element::Type)) {
        auto data_type = attrs[0].type();
        return TYPE_MAP().at(data_type);
    } else if (type_info == typeid(bool)) {
        return attrs[0].b();
    } else if (type_info == typeid(::tensorflow::DataType)) {
        return attrs[0].type();
    } else if (type_info == typeid(::tensorflow::TensorProto)) {
        return attrs[0].tensor();
    } else if (type_info == typeid(::ov::PartialShape)) {
        std::vector<ov::Dimension> dims;
        auto tf_shape = attrs[0].shape();
        for (int i = 0; i < tf_shape.dim_size(); i++) {
            dims.push_back(tf_shape.dim(i).size());
        }
        auto pshape = ov::PartialShape(dims);
        return pshape;
    }

    // type is not supported by decoder
    return {};
#endif
}

size_t OVTFDecoder::get_input_size() const {
#ifdef TF_FE_NO_TF_DEP
    return m_input_size;
#else
    return m_node_def->input_size();
#endif
}

void OVTFDecoder::get_input_node(size_t input_port_idx,
                                    std::string& producer_name,
                                    size_t& producer_output_port_index) const {
#ifdef TF_FE_NO_TF_DEP
#else
    std::string producer_port_name = m_node_def->input(input_port_idx);
    auto delim_pos = producer_port_name.find(':');
    if (delim_pos != std::string::npos) {
        producer_name = producer_port_name.substr(0, delim_pos);
    std::string p_p_idx_str = producer_port_name.substr(delim_pos+1);
        producer_output_port_index = std::stoi(producer_port_name.substr(delim_pos+1));
        return;
    }
    producer_name = producer_port_name;
    producer_output_port_index = 0;
#endif
}

const std::string& OVTFDecoder::get_op_type() const {
#ifdef TF_FE_NO_TF_DEP
    return m_op_type;
#else
    return m_node_def->op();
#endif
}

const std::string& OVTFDecoder::get_op_name() const {
#ifdef TF_FE_NO_TF_DEP
    return m_op_name;
#else
    return m_node_def->name();
#endif
}

#ifdef TF_FE_NO_TF_DEP
void OVTFDecoder::set_next(shared_ptr<OVTFDecoder> next_ptr) {
    m_next_ptr = next_ptr;
}

shared_ptr<OVTFDecoder> OVTFDecoder::get_next() const {
    return m_next_ptr;
}

void OVTFDecoder::add_attr(std::string attr_name, ovtf_attr value) {
    m_attr_map.insert(pair<std::string, ovtf_attr>(attr_name, value));
}

void OVTFDecoder::set_tfnode(shared_ptr<tensorflow::Node> tfnode_ptr) {
    this->m_tfnode_ptr = tfnode_ptr;
}
#else

vector<::tensorflow::AttrValue> OVTFDecoder::decode_attribute_helper(const string& name) const {

    auto attr_map = m_node_def->attr();
    FRONT_END_GENERAL_CHECK(attr_map.contains(name),
                            "An error occurred while parsing the ",
                            name,
                            " attribute of ",
                            this->get_op_type(),
                            "node");
    auto value = m_node_def->attr().at(name);
    return {value};
}
#endif

}  // namespace openvino_tensorflow
}  // namespace tensorflow
