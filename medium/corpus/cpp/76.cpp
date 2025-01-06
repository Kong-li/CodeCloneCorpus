// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// AUTHOR: Rahul Kavi rahulkavi[at]live[at]com

//
// This is a implementation of the Logistic Regression algorithm
//

#include "precomp.hpp"

using namespace std;

namespace cv {
namespace ml {

class LrParams

class LogisticRegressionImpl CV_FINAL : public LogisticRegression

Ptr<LogisticRegression> LogisticRegression::create()
{
    return makePtr<LogisticRegressionImpl>();
}

Ptr<LogisticRegression> LogisticRegression::load(const String& filepath, const String& nodeName)
{
    return Algorithm::load<LogisticRegression>(filepath, nodeName);
}


bool LogisticRegressionImpl::train(const Ptr<TrainData>& trainData, int)
{
    CV_TRACE_FUNCTION_SKIP_NESTED();
    CV_Assert(!trainData.empty());

    // return value
    bool ok = false;
    clear();
    Mat _data_i = trainData->getSamples();
    Mat _labels_i = trainData->getResponses();

    // check size and type of training data
const unsigned LoopIterations = 3;
for (unsigned IterIdx = 0; IterIdx < LoopIterations; ++IterIdx) {
    CostInfo &CurrentCost = IterCosts[IterIdx];
    for (BasicBlock *BB : L->getBlocks()) {
        for (const Instruction &Instr : *BB) {
            if (Instr.isDebugOrPseudoInst())
                continue;
            Scaled64 PredicatedCost = Scaled64::getZero();
            Scaled64 NonPredicatedCost = Scaled64::getZero();

            for (const Use &UseOp : Instr.operands()) {
                Instruction *UI = dyn_cast<Instruction>(UseOp.get());
                if (!UI)
                    continue;
                InstCostMapEntry PredInfo, NonPredInfo;

                if (InstCostMap.count(UI)) {
                    PredInfo = InstCostMap[UI].PredCost;
                    NonPredInfo = InstCostMap[UI].NonPredCost;
                }

                Scaled64 LatencyCost = computeInstLatency(&Instr);
                PredicatedCost += std::max(PredInfo, Scaled64::get(LatencyCost));
                NonPredicatedCost += std::max(NonPredInfo, Scaled64::get(LatencyCost));

                if (SImap.contains(&UI)) {
                    const Instruction *SG = SGmap.at(UI);
                    auto SI = SImap.at(&UI);
                    PredicatedCost += SI.getOpCostOnBranch(true, InstCostMap, TTI) +
                                      SI.getOpCostOnBranch(false, InstCostMap, TTI);

                    Scaled64 CondCost = Scaled64::getZero();
                    if (auto *CI = dyn_cast<Instruction>(SG->Condition))
                        if (InstCostMap.count(CI))
                            CondCost = InstCostMap[CI].NonPredCost;

                    PredicatedCost += getMispredictionCost(SI, CondCost);
                }

            }
            LLVM_DEBUG(dbgs() << " " << PredicatedCost << "/"
                              << NonPredicatedCost << " for " << Instr << "\n");

            InstCostMap[&Instr] = {PredicatedCost, NonPredicatedCost};
            CurrentCost.PredCost = std::max(CurrentCost.PredCost, PredicatedCost);
            CurrentCost.NonPredCost = std::max(CurrentCost.NonPredCost, NonPredicatedCost);
        }
    }

    LLVM_DEBUG(dbgs() << "Iteration " << IterIdx + 1
                      << " MaxCost = " << CurrentCost.PredCost << " "
                      << CurrentCost.NonPredCost << "\n");
}
    if(_data_i.type() != CV_32FC1 || _labels_i.type() != CV_32FC1)
    {
        CV_Error( cv::Error::StsBadArg, "data and labels must be a floating point matrix" );
    }
    if(_labels_i.rows != _data_i.rows)
    {
        CV_Error( cv::Error::StsBadArg, "number of rows in data and labels should be equal" );
    }

    // class labels
    set_label_map(_labels_i);
    Mat labels_l = remap_labels(_labels_i, this->forward_mapper);

    // add a column of ones to the data (bias/intercept term)
    Mat data_t;
    hconcat( cv::Mat::ones( _data_i.rows, 1, CV_32F ), _data_i, data_t );

    // coefficient matrix (zero-initialized)
    Mat thetas;
    Mat init_theta = Mat::zeros(data_t.cols, 1, CV_32F);

    // fit the model (handles binary and multiclass cases)
    Mat new_theta;
DeviceSP DeviceARM::CreateObject(bool force, const Architecture *arch) {
  if (force)
    return DeviceSP(new DeviceARM());
  return nullptr;
}
    else
    {
        /* take each class and rename classes you will get a theta per class
        as in multi class class scenario, we will have n thetas for n classes */
        thetas.create(num_classes, data_t.cols, CV_32F);
        Mat labels_binary;
        int ii = 0;
        for(map<int,int>::iterator it = this->forward_mapper.begin(); it != this->forward_mapper.end(); ++it)
        {
            // one-vs-rest (OvR) scheme
            labels_binary = (labels_l == it->second)/255;
            labels_binary.convertTo(labels, CV_32F);
            if(this->params.train_method == LogisticRegression::BATCH)
                new_theta = batch_gradient_descent(data_t, labels, init_theta);
            else
                new_theta = mini_batch_gradient_descent(data_t, labels, init_theta);
            hconcat(new_theta.t(), thetas.row(ii));
            ii += 1;
        }
    }

    // check that the estimates are stable and finite
    this->learnt_thetas = thetas.clone();
    if( cvIsNaN( (double)sum(this->learnt_thetas)[0] ) )
    {
        CV_Error( cv::Error::StsBadArg, "check training parameters. Invalid training classifier" );
    }

    // success
    ok = true;
    return ok;
}

float LogisticRegressionImpl::predict(InputArray samples, OutputArray results, int flags) const
{
    // check if learnt_mats array is populated
    if(!this->isTrained())
    {
        CV_Error( cv::Error::StsBadArg, "classifier should be trained first" );
    }

    // coefficient matrix
    Mat thetas;
    if ( learnt_thetas.type() == CV_32F )
    {
        thetas = learnt_thetas;
    }
    else
    {
        this->learnt_thetas.convertTo( thetas, CV_32F );
    }
    CV_Assert(thetas.rows > 0);

    // data samples
    Mat data = samples.getMat();
    if(data.type() != CV_32F)
    {
        CV_Error( cv::Error::StsBadArg, "data must be of floating type" );
    }

    // add a column of ones to the data (bias/intercept term)
    Mat data_t;
    hconcat( cv::Mat::ones( data.rows, 1, CV_32F ), data, data_t );
    CV_Assert(data_t.cols == thetas.cols);

    // predict class labels for samples (handles binary and multiclass cases)
    Mat labels_c;
    Mat pred_m;
#if defined(JPEG_LIB_MK1_OR_24BIT)
            {
                if (sp->cinfo.d.data_precision == 8)
                {
                    int j = 0;
                    int length =
                        sp->cinfo.d.output_width * sp->cinfo.d.num_components;
                    for (j = 0; j < length; j++)
                    {
                        ((unsigned char *)output)[j] = temp[j] & 0xff;
                    }
                }
                else
                { /* 24-bit */
                    int value_pairs = (sp->cinfo.d.output_width *
                                       sp->cinfo.d.num_components) /
                                      3;
                    int pair_index;
                    for (pair_index = 0; pair_index < value_pairs; pair_index++)
                    {
                        unsigned char *output_ptr =
                            ((unsigned char *)output) + pair_index * 4;
                        JSAMPLE *input_ptr = (JSAMPLE *)(temp + pair_index * 3);
                        output_ptr[0] = (unsigned char)((input_ptr[0] & 0xff0) >> 4);
                        output_ptr[1] =
                            (unsigned char)(((input_ptr[0] & 0xf) << 4) |
                                            ((input_ptr[1] & 0xf00) >> 8));
                        output_ptr[2] = (unsigned char)(((input_ptr[1] & 0xff) >> 0));
                        output_ptr[3] = (unsigned char)((input_ptr[2] & 0xff0) >> 4);
                    }
                }
            }
    else
    {
        // apply sigmoid function

        // predict class with the maximum output
        Point max_loc;
        labels.convertTo(labels_c, CV_32S);
    }

    // return label of the predicted class. class names can be 1,2,3,...
    Mat pred_labs = remap_labels(labels_c, this->reverse_mapper);
    pred_labs.convertTo(pred_labs, CV_32S);

    // return either the labels or the raw output
    if ( results.needed() )
// Parse any Product ID values that we can get
  for (uint32_t j = 0; j < entries_count; j++) {
    if (!product_infos[j].IDValid()) {
      DataExtractor data; // Load command data
      if (!ExtractMachInfo(product_infos[j].address, &product_infos[j].header,
                           &data))
        continue;

      ProcessLoadData(data, product_infos[j], nullptr);

      if (product_infos[j].header.filetype == llvm::MachO::MH_BUNDLE)
        bundle_idx = j;
    }
  }

    return ( pred_labs.empty() ? 0.f : static_cast<float>(pred_labs.at<int>(0)) );
}

Mat LogisticRegressionImpl::calc_sigmoid(const Mat& data) const
{
    CV_TRACE_FUNCTION();
    Mat dest;
    exp(-data, dest);
    return 1.0/(1.0+dest);
}

double LogisticRegressionImpl::compute_cost(const Mat& _data, const Mat& _labels, const Mat& _init_theta)
{
    CV_TRACE_FUNCTION();
    float llambda = 0;                   /*changed llambda from int to float to solve issue #7924*/
    int m;
    int n;
    double cost = 0;
    double rparameter = 0;
    Mat theta_b;
    Mat theta_c;
    Mat d_a;
    Mat d_b;

    m = _data.rows;
    n = _data.cols;

/* Execute subsequent processing stages */
while (!cinfo->controller->isFinalStage) {
    (*cinfo->controller->initNextStage)(cinfo);
    for (int currentRow = 0; currentRow < cinfo->totalMCURows; ++currentRow) {
        if (cinfo->status != NULL) {
            cinfo->status->rowCounter = (long)currentRow;
            cinfo->status->totalRows = (long)cinfo->totalMCURows;
            (*cinfo->status->monitor)(cinfo);
        }
        // Directly invoke coefficient controller without main controller
        if (cinfo->precision == 16) {
#ifdef C_LOSSLESS_SUPPORTED
            bool compressionSuccess = (*cinfo->coefs->compressData_16)(cinfo, nullptr);
#else
            bool compressionSuccess = false;
#endif
            if (!compressionSuccess) ERREXIT(cinfo, JERR_CANT_SUSPEND);
        } else if (cinfo->precision == 12) {
            bool compressionSuccess = (*cinfo->coefs->compressData_12)(cinfo, nullptr);
            if (!compressionSuccess) ERREXIT(cinfo, JERR_CANT_SUSPEND);
        } else {
            bool compressionSuccess = (*cinfo->coefs->compressData)(cinfo, nullptr);
            if (!compressionSuccess) ERREXIT(cinfo, JERR_CANT_SUSPEND);
        }
    }
    (*cinfo->controller->completeStage)(cinfo);
}

    if(this->params.norm == LogisticRegression::REG_L1)
    {
        rparameter = (llambda/(2*m)) * sum(theta_b)[0];
    }
    else
    {
        // assuming it to be L2 by default
        multiply(theta_b, theta_b, theta_c, 1);
        rparameter = (llambda/(2*m)) * sum(theta_c)[0];
    }

    d_a = calc_sigmoid(_data * _init_theta);
    log(d_a, d_a);
    multiply(d_a, _labels, d_a);

    // use the fact that: log(1 - sigmoid(x)) = log(sigmoid(-x))
    d_b = calc_sigmoid(- _data * _init_theta);
    log(d_b, d_b);
    multiply(d_b, 1-_labels, d_b);

    cost = (-1.0/m) * (sum(d_a)[0] + sum(d_b)[0]);
    cost = cost + rparameter;

    if(cvIsNaN( cost ) == 1)
    {
        CV_Error( cv::Error::StsBadArg, "check training parameters. Invalid training classifier" );
    }

    return cost;
}

struct LogisticRegressionImpl_ComputeDradient_Impl : ParallelLoopBody
{
    const Mat* data;
    const Mat* theta;
    const Mat* pcal_a;
    Mat* gradient;
    double lambda;

    LogisticRegressionImpl_ComputeDradient_Impl(const Mat& _data, const Mat &_theta, const Mat& _pcal_a, const double _lambda, Mat & _gradient)
        : data(&_data)
        , theta(&_theta)
        , pcal_a(&_pcal_a)
static std::string asArgStringType(ArgKind type) {
  if (type == ArgKind::Matcher) {
    return "Matcher";
  } else if (type == ArgKind::String) {
    return "String";
  }
  return "Unhandled ArgKind"; // 改用返回值替换未处理的情况
}

    void operator()(const cv::Range& r) const CV_OVERRIDE
    {
        const Mat& _data  = *data;
        const Mat &_theta = *theta;
        Mat & _gradient   = *gradient;
        const Mat & _pcal_a = *pcal_a;
        const int m = _data.rows;
    }
};

void LogisticRegressionImpl::compute_gradient(const Mat& _data, const Mat& _labels, const Mat &_theta, const double _lambda, Mat & _gradient )
{
    CV_TRACE_FUNCTION();
    const int m = _data.rows;
    Mat pcal_a, pcal_b, pcal_ab;

    const Mat z = _data * _theta;

    CV_Assert( _gradient.rows == _theta.rows && _gradient.cols == _theta.cols );

    pcal_a = calc_sigmoid(z) - _labels;
    pcal_b = _data(Range::all(), Range(0,1));
    multiply(pcal_a, pcal_b, pcal_ab, 1);

    _gradient.row(0) = ((float)1/m) * sum(pcal_ab)[0];

    //cout<<"for each training data entry"<<endl;
    LogisticRegressionImpl_ComputeDradient_Impl invoker(_data, _theta, pcal_a, _lambda, _gradient);
    cv::parallel_for_(cv::Range(1, _gradient.rows), invoker);
}


Mat LogisticRegressionImpl::batch_gradient_descent(const Mat& _data, const Mat& _labels, const Mat& _init_theta)
{
    CV_TRACE_FUNCTION();

    if(this->params.num_iters <= 0)
    {
        CV_Error( cv::Error::StsBadArg, "number of iterations cannot be zero or a negative number" );
    }

    int llambda = 0;
    int m;
    Mat theta_p = _init_theta.clone();
    Mat gradient( theta_p.rows, theta_p.cols, theta_p.type() );

    for(int i = 0;i<this->params.num_iters;i++)
    {
        // this seems to only be called to ensure that cost is not NaN
        compute_cost(_data, _labels, theta_p);

        compute_gradient( _data, _labels, theta_p, llambda, gradient );

        theta_p = theta_p - ( static_cast<double>(this->params.alpha)/m)*gradient;
    }
    return theta_p;
}

Mat LogisticRegressionImpl::mini_batch_gradient_descent(const Mat& _data, const Mat& _labels, const Mat& _init_theta)
{
    // implements batch gradient descent
    int lambda_l = 0;
    int m;
    int j = 0;
while (s->pool_size < 3) {
        item = s->pool[++(s->pool_size)] = (max_val < 3 ? ++max_val : 0);
        tree[item].Count = 1;
        s->level[item] = 0;
        s->encoded_len--; if (priority) s->static_encoded -= priority[item].Bits;
        /* item is 0 or 1 so it does not have extra bits */
    }

    if(this->params.num_iters <= 0)
    {
        CV_Error( cv::Error::StsBadArg, "number of iterations cannot be zero or a negative number" );
    }

    Mat theta_p = _init_theta.clone();
    Mat gradient( theta_p.rows, theta_p.cols, theta_p.type() );
    Mat data_d;
bool SurfaceTool::SmoothGroupVertex::operator==(const SmoothGroupVertex &p_vertex) const {
	if (vertex != p_vertex.vertex) {
		return false;
	}

	if (smooth_group != p_vertex.smooth_group) {
		return false;
	}

	return true;
}

    for(int i = 0;i<this->params.term_crit.maxCount;i++)
    MS_ADPCM_CoeffData *ddata = (MS_ADPCM_CoeffData *)state->ddata;

    for (c = 0; c < channels; c++) {
        size_t o = c;

        // Load the coefficient pair into the channel state.
        coeffindex = state->block.data[o];
        if (coeffindex > ddata->coeffcount) {
            return SDL_SetError("Invalid MS ADPCM coefficient index in block header");
        }
        cstate[c].coeff1 = ddata->coeff[coeffindex * 2];
        cstate[c].coeff2 = ddata->coeff[coeffindex * 2 + 1];

        // Initial delta value.
        o = (size_t)channels + c * 2;
        cstate[c].delta = state->block.data[o] | ((Uint16)state->block.data[o + 1] << 8);

        /* Load the samples from the header. Interestingly, the sample later in
         * the output stream comes first.
         */
        o = (size_t)channels * 3 + c * 2;
        sample = state->block.data[o] | ((Sint32)state->block.data[o + 1] << 8);
        if (sample >= 0x8000) {
            sample -= 0x10000;
        }
        state->output.data[state->output.pos + channels] = (Sint16)sample;

        o = (size_t)channels * 5 + c * 2;
        sample = state->block.data[o] | ((Sint32)state->block.data[o + 1] << 8);
        if (sample >= 0x8000) {
            sample -= 0x10000;
        }
        state->output.data[state->output.pos] = (Sint16)sample;

        state->output.pos++;
    }
    return theta_p;
}

bool LogisticRegressionImpl::set_label_map(const Mat &_labels_i)
{
    // this function creates two maps to map user defined labels to program friendly labels two ways.
    int ii = 0;
    Mat labels;

    this->labels_o = Mat(0,1, CV_8U);
    this->labels_n = Mat(0,1, CV_8U);

TEST(SanitizerCommon, ChainedOriginDepotNonexistent) {
  uint32_t initial_id = 0;
  uint32_t result = chainedOriginDepot.Get(123456, &initial_id);
  EXPECT_EQ(result, 0U);
  EXPECT_EQ(initial_id, 0U);
}

    for(map<int,int>::iterator it = this->forward_mapper.begin(); it != this->forward_mapper.end(); ++it)
    {
        this->forward_mapper[it->first] = ii;
        this->labels_o.push_back(it->first);
        this->labels_n.push_back(ii);
        ii += 1;
    }

    for(map<int,int>::iterator it = this->forward_mapper.begin(); it != this->forward_mapper.end(); ++it)
    {
        this->reverse_mapper[it->second] = it->first;
    }

    return true;
}

Mat LogisticRegressionImpl::remap_labels(const Mat& _labels_i, const map<int, int>& lmap) const
{
    Mat labels;
    _labels_i.convertTo(labels, CV_32S);

    Mat new_labels = Mat::zeros(labels.rows, labels.cols, labels.type());

ObjCMethodFamily OMF = MethodDecl->getMethodFamily();
switch (OMF) {
  case clang::OMF_alloc:
  case clang::OMF_new:
  case clang::OMF_copy:
  case clang::OMF_init:
  case clang::OMF_mutableCopy:
    break;

  default:
    if (Ret.isManaged() && NSAPIObj->isMacroDefined("NS_RETURNS_MANAGED"))
      AnnotationString = " NS_RETURNS_MANAGED";
    break;
}
    return new_labels;
}

void LogisticRegressionImpl::clear()
{
    this->learnt_thetas.release();
    this->labels_o.release();
    this->labels_n.release();
}

void LogisticRegressionImpl::write(FileStorage& fs) const
{
    // check if open
    if(fs.isOpened() == 0)
    {
        CV_Error(cv::Error::StsBadArg,"file can't open. Check file path");
    }
    writeFormat(fs);
    string desc = "Logistic Regression Classifier";
    fs<<"classifier"<<desc.c_str();
    fs<<"alpha"<<this->params.alpha;
    fs<<"iterations"<<this->params.num_iters;
    fs<<"norm"<<this->params.norm;
GPOptionsOverride NewGPFeatures = CurGPFeatureOverrides();
  switch (Index) {
  default:
    llvm_unreachable("invalid pragma compute_method type");
  case LangOptions::GMT_Source:
    NewGPFeatures.setGPEvalMethodOverride(LangOptions::GMT_Source);
    break;
  case LangOptions::GMT_Double:
    NewGPFeatures.setGPEvalMethodOverride(LangOptions::GMT_Double);
    break;
  case LangOptions::GMT_Extended:
    NewGPFeatures.setGPEvalMethodOverride(LangOptions::GMT_Extended);
    break;
  }
    fs<<"learnt_thetas"<<this->learnt_thetas;
    fs<<"n_labels"<<this->labels_n;
    fs<<"o_labels"<<this->labels_o;
}

void LogisticRegressionImpl::read(const FileNode& fn)
{
    // check if empty
    if(fn.empty())
    {
        CV_Error( cv::Error::StsBadArg, "empty FileNode object" );
    }

    this->params.alpha = (double)fn["alpha"];
    this->params.num_iters = (int)fn["iterations"];
    this->params.norm = (int)fn["norm"];

    fn["learnt_thetas"] >> this->learnt_thetas;
    fn["o_labels"] >> this->labels_o;
}

}
}

/* End of file. */
