#pragma once

#include <dlib/base64.h>
#include <dlib/compress_stream.h>
#include <dlib/dnn.h>

namespace face_detector
{
    // A 5x5 conv layer that does 2x downsampling
    template <long num_filters, typename SUBNET>
    using con5d = dlib::con<num_filters,5,5,2,2,SUBNET>;
    // A 3x3 conv layer that doesn't do any downsampling
    template <long num_filters, typename SUBNET>
    using con3  = dlib::con<num_filters,3,3,1,1,SUBNET>;

    // Now we can define the 8x downsampling block in terms of conv5d blocks.
    // We also use relu and batch normalization in the standard way.
    template <template<typename> class BN, typename SUBNET>
    using downsampler = dlib::relu<BN<con5d<32, dlib::relu<BN<con5d<32, dlib::relu<BN<con5d<32,SUBNET>>>>>>>>>;

    // The rest of the network will be 3x3 conv layers with batch normalization
    // and relu.  So we define the 3x3 block we will use here.
    template <template<typename> class BN, typename SUBNET>
    using rcon3  = dlib::relu<BN<con3<32,SUBNET>>>;

    // Finally, we define the entire network.
    // The special input_rgb_image_pyramid layer causes the network to operate
    // over a spatial pyramid, making the detector scale invariant.
    template<template<typename> class BN>
    using net_type = dlib::loss_mmod<
        dlib::con<1, 6, 6, 1, 1,
        rcon3<BN, rcon3<BN, rcon3<BN,
        downsampler<BN,
        dlib::input_rgb_image_pyramid<dlib::pyramid_down<6>>
    >>>>>>;

    // some useful aliase for train and infer modes
    using train = net_type<dlib::bn_con>;
    using infer = net_type<dlib::affine>;
}

// HOW TO GENERATE THE COMPRESSED STREAM:
/*
    dlib::deserialize(model_path) >> net;
    std::ostringstream sout;
    std::istringstream sin;
    dlib::base64 base64_coder;
    dlib::compress_stream::kernel_1ea compressor;
    // put the data into ostream sout
    dlib::serialize(net, sout);
    // put the data into istream sin
    sin.str(sout.str());
    sout.str("");
    // compress the data
    compressor.compress(sin, sout);
    sin.clear();
    sin.str(sout.str());
    sout.str("");
    // encode the data into base64
    base64_coder.encode(sin, sout);
    // print the data on the terminal
    std::cout << sout.str() << std::endl;
*/

const std::string get_serialized_mmod_face_detector();
