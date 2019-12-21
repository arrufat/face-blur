#include <iostream>

#include <dlib/cmd_line_parser.h>
#include <dlib/dir_nav.h>
#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>

#include "detector_model.h"

const std::string img_ext = ".jpg .jpeg .png .gif .JPG .JPEG .PNG .GIF";

void paste(
    dlib::matrix<dlib::rgb_pixel>& img,
    const dlib::rectangle& rect,
    const dlib::matrix<dlib::rgb_pixel>& chip
)
{
    long top = rect.top() < 0 ? 0 : rect.top();
    long bottom = rect.bottom() >= img.nr() ? img.nr() : rect.bottom();
    long left = rect.left() < 0 ? 0 : rect.left();
    long right = rect.right() >= img.nc() ? img.nc() : rect.right();
    for (long i = top; i < bottom; ++i)
    {
        for (long j = left; j < right; ++j)
        {
            img(i, j) = chip(i - top, j - left);
        }
    }
}

int main(int argc, char** argv) try
{
    dlib::command_line_parser parser;
    parser.add_option("fast", "use a faster, less accurate face detector");
    parser.add_option("sigma", "size of the gaussian blur kernel (default: 3)", 1);
    parser.set_group_name("Help Options");
    parser.add_option("h", "alias for --help");
    parser.add_option("help", "display this message and exit");
    parser.parse(argc, argv);

    std::cout << "Face Blur" << std::endl;

    if (argc == 1 || parser.option("h") || parser.option("help"))
    {
        parser.print_options();
        return EXIT_SUCCESS;
    }

    face_detector::infer dnn_detector;
    std::istringstream sin(get_serialized_mmod_face_detector());
    deserialize(dnn_detector, sin);
    auto hog_detector = dlib::get_frontal_face_detector();

    const double sigma = get_option(parser, "sigma", 3);

    std::vector<dlib::file> files;
    for (size_t i = 0; i < parser.number_of_arguments(); ++i)
    {
        std::cout << parser[i] << std::endl;
        try
        {
            dlib::directory dir(parser[i]);
            files = dlib::get_files_in_directory_tree(parser[i], dlib::match_endings(img_ext));
        }
        catch (dlib::directory::dir_not_found& ) { }
        try
        {
            dlib::file file(parser[i]);
            files.push_back(file);
        } catch (dlib::file::file_not_found& ) { }
    }

    if (files.size() == 0)
    {
        std::cout << "no images found" << std::endl;
        return EXIT_SUCCESS;
    }


    std::cout << "processing " << files.size() << " images" << std::endl;

    dlib::image_window win;
    win.set_title("Face Blur");
    dlib::matrix<dlib::rgb_pixel> img, face_chip, face_blur, face_final;
    for (const auto& file : files)
    {
        dlib::load_image(img, file.full_name());
        win.set_image(img);
        std::vector<dlib::rectangle> dets;
        if (parser.option("fast"))
        {
            dets = hog_detector(img);
        }
        else
        {
            for (const auto& det : dnn_detector(img))
            {
                dets.push_back(det.rect);
            }
        }
        for (auto&& det : dets)
        {

            dlib::extract_image_chip(img, det, face_chip);
            face_blur.set_size(face_chip.nr() / 8, face_chip.nc() / 8);
            dlib::resize_image(face_chip, face_blur, dlib::interpolate_nearest_neighbor());
            face_final.set_size(face_chip.nr(), face_chip.nc());
            dlib::resize_image(face_blur, face_final, dlib::interpolate_nearest_neighbor());

            // // extract a big area around the detector to prevent border effects when blurring
            // const auto p = 2;
            // const auto box = dlib::centered_rect(det.rect, det.rect.width() * p, det.rect.height() * p);
            // dlib::extract_image_chip(img, box, face_chip);
            // dlib::extract_image_chip(img, box, face_chip);
            // dlib::gaussian_blur(face_chip, face_blur, sigma);
            // // get the blurred face inside the blurred chip
            // const auto final_box = dlib::centered_rect(
            //     face_blur.nc() / 2,
            //     face_blur.nr() / 2,
            //     det.rect.width(),
            //     det.rect.height());
            // dlib::extract_image_chip(face_blur, final_box, face_final);

            paste(img, det, face_final);
        }
        win.clear_overlay();
        win.set_image(img);
        dlib::save_png(img, "blurred.png");
        // win.wait_until_closed();
        // std::cin.get();
    }

    return EXIT_SUCCESS;
}
catch (std::exception& e)
{
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
}
