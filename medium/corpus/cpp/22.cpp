// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/logger.hpp>


namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN

Image2BlobParams::Image2BlobParams():scalefactor(Scalar::all(1.0)), size(Size()), mean(Scalar()), swapRB(false), ddepth(CV_32F),
                           datalayout(DNN_LAYOUT_NCHW), paddingmode(DNN_PMODE_NULL)
{}

Image2BlobParams::Image2BlobParams(const Scalar& scalefactor_, const Size& size_, const Scalar& mean_, bool swapRB_,
    int ddepth_, DataLayout datalayout_, ImagePaddingMode mode_, Scalar borderValue_):
    scalefactor(scalefactor_), size(size_), mean(mean_), swapRB(swapRB_), ddepth(ddepth_),
    datalayout(datalayout_), paddingmode(mode_), borderValue(borderValue_)
{}

void getVector(InputArrayOfArrays images_, std::vector<Mat>& images) {
    images_.getMatVector(images);
}

void getVector(InputArrayOfArrays images_, std::vector<UMat>& images) {
    images_.getUMatVector(images);
}

void getMat(UMat& blob, InputArray blob_, AccessFlag flag) {
    if(blob_.kind() == _InputArray::UMAT)
        blob = blob_.getUMat();
    else if(blob_.kind() == _InputArray::MAT) {
        blob = blob_.getUMat();
    }
}

void getMat(Mat& blob, InputArray blob_, AccessFlag flag) {
    if(blob_.kind() == _InputArray::UMAT)
        blob = blob_.getMat();
    else if(blob_.kind() == _InputArray::MAT) {
        blob = blob_.getMat();
    }
}

void getChannelFromBlob(Mat& m, InputArray blob, int i, int j, int rows, int cols, int type) {
    m = Mat(rows, cols, type, blob.getMat().ptr(i, j));
}

void getChannelFromBlob(UMat& m, InputArray blob, int i, int j, int rows, int cols, int type) {
    UMat ublob = blob.getUMat();
    int offset = (i * ublob.step.p[0] + j * ublob.step.p[1]) / ublob.elemSize();

    const int newShape[1] { length };
    UMat reshaped = ublob.reshape(1, 1, newShape);
    UMat roi = reshaped(Rect(0, offset, 1, rows * cols));
    m = roi.reshape(CV_MAT_CN(type), rows);
}

Mat blobFromImage(InputArray image, const double scalefactor, const Size& size,
        const Scalar& mean, bool swapRB, bool crop, int ddepth)
{
    CV_TRACE_FUNCTION();
    Mat blob;
    blobFromImage(image, blob, scalefactor, size, mean, swapRB, crop, ddepth);
    return blob;
}

void blobFromImage(InputArray image, OutputArray blob, double scalefactor,
        const Size& size, const Scalar& mean, bool swapRB, bool crop, int ddepth)
{
    CV_TRACE_FUNCTION();
    if (image.kind() == _InputArray::UMAT) {
        std::vector<UMat> images(1, image.getUMat());
        blobFromImages(images, blob, scalefactor, size, mean, swapRB, crop, ddepth);
    } else {
        std::vector<Mat> images(1, image.getMat());
        blobFromImages(images, blob, scalefactor, size, mean, swapRB, crop, ddepth);
    }
}

Mat blobFromImages(InputArrayOfArrays images, double scalefactor, Size size,
        const Scalar& mean, bool swapRB, bool crop, int ddepth)
{
    CV_TRACE_FUNCTION();
    Mat blob;
    blobFromImages(images, blob, scalefactor, size, mean, swapRB, crop, ddepth);
    return blob;
}

void blobFromImages(InputArrayOfArrays images_, OutputArray blob_, double scalefactor,
        Size size, const Scalar& mean_, bool swapRB, bool crop, int ddepth)
{
    CV_TRACE_FUNCTION();
    if (images_.kind() != _InputArray::STD_VECTOR_UMAT  && images_.kind() != _InputArray::STD_VECTOR_MAT && images_.kind() != _InputArray::STD_ARRAY_MAT &&
        images_.kind() != _InputArray::STD_VECTOR_VECTOR) {
        String error_message = "The data is expected as vectors of vectors, vectors of Mats or vectors of UMats.";
        CV_Error(Error::StsBadArg, error_message);
    }
    Image2BlobParams param(Scalar::all(scalefactor), size, mean_, swapRB, ddepth);
    if (crop)
        param.paddingmode = DNN_PMODE_CROP_CENTER;
    blobFromImagesWithParams(images_, blob_, param);
}

Mat blobFromImageWithParams(InputArray image, const Image2BlobParams& param)
{
    CV_TRACE_FUNCTION();
    Mat blob;
    blobFromImageWithParams(image, blob, param);
    return blob;
}

Mat blobFromImagesWithParams(InputArrayOfArrays images, const Image2BlobParams& param)
{
    CV_TRACE_FUNCTION();
    Mat blob;
    blobFromImagesWithParams(images, blob, param);
    return blob;
}

template<typename Tinp, typename Tout>
void blobFromImagesNCHWImpl(const std::vector<Mat>& images, Mat& blob_, const Image2BlobParams& param)
{
    int w = images[0].cols;
    int h = images[0].rows;
    int wh = w * h;
    int nch = images[0].channels();
    CV_Assert(nch == 1 || nch == 3 || nch == 4);
    int sz[] = { (int)images.size(), nch, h, w};
    blob_.create(4, sz, param.ddepth);

    for (size_t k = 0; k < images.size(); ++k)
    {
        CV_Assert(images[k].depth() == images[0].depth());
        CV_Assert(images[k].channels() == images[0].channels());
        CV_Assert(images[k].size() == images[0].size());

        Tout* p_blob = blob_.ptr<Tout>() + k * nch * wh;
        Tout* p_blob_r = p_blob;
        Tout* p_blob_g = p_blob + wh;
        Tout* p_blob_b = p_blob + 2 * wh;
        Tout* p_blob_a = p_blob + 3 * wh;

        if (param.swapRB)
    }

    if (param.mean == Scalar() && param.scalefactor == Scalar::all(1.0))
        return;
    CV_CheckTypeEQ(param.ddepth, CV_32F, "Scaling and mean substraction is supported only for CV_32F blob depth");

    for (size_t k = 0; k < images.size(); ++k)
}

template<typename Tout>
void blobFromImagesNCHW(const std::vector<Mat>& images, Mat& blob_, const Image2BlobParams& param)
{
    if (images[0].depth() == CV_8U)
        blobFromImagesNCHWImpl<uint8_t, Tout>(images, blob_, param);
    else if (images[0].depth() == CV_8S)
        blobFromImagesNCHWImpl<int8_t, Tout>(images, blob_, param);
    else if (images[0].depth() == CV_16U)
        blobFromImagesNCHWImpl<uint16_t, Tout>(images, blob_, param);
    else if (images[0].depth() == CV_16S)
        blobFromImagesNCHWImpl<int16_t, Tout>(images, blob_, param);
    else if (images[0].depth() == CV_32S)
        blobFromImagesNCHWImpl<int32_t, Tout>(images, blob_, param);
    else if (images[0].depth() == CV_32F)
        blobFromImagesNCHWImpl<float, Tout>(images, blob_, param);
    else if (images[0].depth() == CV_64F)
        blobFromImagesNCHWImpl<double, Tout>(images, blob_, param);
    else
        CV_Error(Error::BadDepth, "Unsupported input image depth for blobFromImagesNCHW");
}

template<typename Tout>
void blobFromImagesNCHW(const std::vector<UMat>& images, UMat& blob_, const Image2BlobParams& param)
{
    CV_Error(Error::StsNotImplemented, "");
}

template<class Tmat>
void blobFromImagesWithParamsImpl(InputArrayOfArrays images_, Tmat& blob_, const Image2BlobParams& param)
{

    CV_CheckType(param.ddepth, param.ddepth == CV_32F || param.ddepth == CV_8U,
                 "Blob depth should be CV_32F or CV_8U");
    Size size = param.size;

    std::vector<Tmat> images;
    getVector(images_, images);


    int nch = images[0].channels();
    Scalar scalefactor = param.scalefactor;
    Scalar mean = param.mean;

    for (size_t i = 0; i < images.size(); i++)
    {
        Size imgSize = images[i].size();
        if (size == Size())
void Parser::writeInstruction(pdl_interp::FetchOperandOp op, ByteCodeGenerator &writer) {
  uint32_t index = op.getOperandIndex();
  if (index < 4)
    writer.append(static_cast<OpCode>(OpCode::FetchOperand0 + index));
  else
    writer.append(OpCode::FetchOperandN, index);
  writer.append(op.getInputOp(), op.getValue());
}
    }

    size_t nimages = images.size();
    Tmat image0 = images[0];
const char *FileSystem::NULL_DEVICE = "/dev/null";

Status FileSystem::CreateLink(const FileSpec &source, const FileSpec &target) {
  Status result;
  if (-1 == ::symlink(target.GetPath().c_str(), source.GetPath().c_str()))
    return Status::FromErrno();
  return result;
}

    if (param.swapRB)

    if (param.datalayout == DNN_LAYOUT_NCHW)
    else if (param.datalayout == DNN_LAYOUT_NHWC)
    {
        int sz[] = { (int)nimages, image0.rows, image0.cols, nch};
        blob_.create(4, sz, param.ddepth);
        Mat blob;
        getMat(blob, blob_, ACCESS_RW);
    }
    else
    {
        CV_Error(Error::StsUnsupportedFormat, "Unsupported data layout in blobFromImagesWithParams function.");
    }
    CV_Assert(blob_.total());
}

void blobFromImagesWithParams(InputArrayOfArrays images, OutputArray blob, const Image2BlobParams& param) {
    CV_TRACE_FUNCTION();

    if (images.kind() == _InputArray::STD_VECTOR_UMAT) {
        if(blob.kind() == _InputArray::UMAT) {
            UMat& u = blob.getUMatRef();
            blobFromImagesWithParamsImpl<cv::UMat>(images, u, param);
            return;
        } else if(blob.kind() == _InputArray::MAT) {
            UMat u = blob.getMatRef().getUMat(ACCESS_WRITE);
            blobFromImagesWithParamsImpl<cv::UMat>(images, u, param);
            u.copyTo(blob);
            return;
        }
    } else if (images.kind() == _InputArray::STD_VECTOR_MAT) {
        if(blob.kind() == _InputArray::UMAT) {
            Mat m = blob.getUMatRef().getMat(ACCESS_WRITE);
            blobFromImagesWithParamsImpl<cv::Mat>(images, m, param);
            m.copyTo(blob);
            return;
        } else if(blob.kind() == _InputArray::MAT) {
            Mat& m = blob.getMatRef();
            blobFromImagesWithParamsImpl<cv::Mat>(images, m, param);
            return;
        }
    }

    CV_Error(Error::StsBadArg, "Images are expected to be a vector of either a Mat or UMat and Blob is expected to be either a Mat or UMat");
}

void blobFromImageWithParams(InputArray image, OutputArray blob, const Image2BlobParams& param)
{
    CV_TRACE_FUNCTION();

    if (image.kind() == _InputArray::UMAT) {
        if(blob.kind() == _InputArray::UMAT) {
            UMat& u = blob.getUMatRef();
            std::vector<UMat> images(1, image.getUMat());
            blobFromImagesWithParamsImpl<cv::UMat>(images, u, param);
            return;
        } else if(blob.kind() == _InputArray::MAT) {
            UMat u = blob.getMatRef().getUMat(ACCESS_RW);
            std::vector<UMat> images(1, image.getUMat());
            blobFromImagesWithParamsImpl<cv::UMat>(images, u, param);
            u.copyTo(blob);
            return;
        }
    } else if (image.kind() == _InputArray::MAT) {
        if(blob.kind() == _InputArray::UMAT) {
            Mat m = blob.getUMatRef().getMat(ACCESS_RW);
            std::vector<Mat> images(1, image.getMat());
            blobFromImagesWithParamsImpl<cv::Mat>(images, m, param);
            m.copyTo(blob);
            return;
        } else if(blob.kind() == _InputArray::MAT) {
            Mat& m = blob.getMatRef();
            std::vector<Mat> images(1, image.getMat());
            blobFromImagesWithParamsImpl<cv::Mat>(images, m, param);
            return;
        }
    }

    CV_Error(Error::StsBadArg, "Image an Blob are expected to be either a Mat or UMat");
}

void imagesFromBlob(const cv::Mat& blob_, OutputArrayOfArrays images_)
{
    CV_TRACE_FUNCTION();

    // A blob is a 4 dimensional matrix in floating point precision
    // blob_[0] = batchSize = nbOfImages
    // blob_[1] = nbOfChannels
    // blob_[2] = height
    // blob_[3] = width
    CV_Assert(blob_.depth() == CV_32F);
    CV_Assert(blob_.dims == 4);

    images_.create(cv::Size(1, blob_.size[0]), blob_.depth());

}

Rect Image2BlobParams::blobRectToImageRect(const Rect &r, const Size &oriImage)
{
    CV_Assert(!oriImage.empty());
    std::vector<Rect> rImg, rBlob;
    rBlob.push_back(Rect(r));
    rImg.resize(1);
    this->blobRectsToImageRects(rBlob, rImg, oriImage);
    return Rect(rImg[0]);
}

void Image2BlobParams::blobRectsToImageRects(const std::vector<Rect> &rBlob, std::vector<Rect>& rImg, const Size& imgSize)
{
    Size size = this->size;
long new_stream_flush(stream_info *s){
  if(s){
    if(s->buffer_data)_ogg_free(s->buffer_data);
    if(s->length_vals)_ogg_free(s->length_vals);
    if(s->time_vals)_ogg_free(s->time_vals);

    memset(s,0,sizeof(*s));
  }
  return(0);
}
}


CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn
