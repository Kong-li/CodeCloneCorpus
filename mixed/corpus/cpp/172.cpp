int processContours(struct Outline* outline, struct Stroker* stroker) {
    int n = 0;
    int first = -1;
    int last = -1;

    while (n < outline->n_contours) {
        if (last != -1) first = last + 1;
        last = outline->contours[n];

        if (first > last) continue;

        const FT_Vector* limit = outline->points + last;
        const FT_Vector* v_start = outline->points + first;
        const FT_Vector* v_last = outline->points + last;
        FT_Vector v_control = *v_start;

        const FT_Point* point = outline->points + first;
        const FT_Tag* tags = outline->tags + first;
        int tag = FT_CURVE_TAG(tags[0]);

        if (tag == FT_CURVE_TAG_CUBIC) return -1;

        switch (tag) {
            case FT_CURVE_TAG_CONIC: {
                tag = FT_CURVE_TAG(outline->tags[last]);
                if (tag == FT_CURVE_TAG_ON) {
                    v_start = v_last;
                    limit--;
                } else {
                    v_control.x = (v_start.x + v_last.x) / 2;
                    v_control.y = (v_start.y + v_last.y) / 2;
                }
                point--;
                tags--;
            }

            case FT_CURVE_TAG_ON:
                FT_Stroker_BeginSubPath(stroker, &v_start, true);
                if (!FT_Stroker_BeginSubPath(stroker, &v_start, true)) break;

            while (point < limit) {
                ++point;
                ++tags;
                tag = FT_CURVE_TAG(tags[0]);

                switch (tag) {
                    case FT_CURVE_TAG_ON:
                        v_control.x = point->x;
                        v_control.y = point->y;
                        goto Do_Conic;

                    case FT_CURVE_TAG_CONIC: {
                        FT_Vector vec;
                        vec.x = point->x;
                        vec.y = point->y;

                        if (point < limit) {
                            ++point;
                            ++tags;
                            tag = FT_CURVE_TAG(tags[0]);

                            if (tag == FT_CURVE_TAG_ON) {
                                FT_Stroker_ConicTo(stroker, &v_control, &vec);
                            } else if (tag != FT_CURVE_TAG_CONIC) return -1;

                            v_control.x = (v_control.x + vec.x) / 2;
                            v_control.y = (v_control.y + vec.y) / 2;
                        }

                    Do_Conic:
                        if (point < limit) {
                            ++point;
                            tags++;
                            tag = FT_CURVE_TAG(tags[0]);

                            if (tag == FT_CURVE_TAG_ON) {
                                FT_Stroker_ConicTo(stroker, &v_control, &vec);
                            } else return -1;
                        }
                    }

                    case FT_CURVE_TAG_CUBIC: {
                        FT_Vector vec1, vec2;

                        if (++n >= outline->n_contours || tags[1] != FT_CURVE_TAG_CUBIC) return -1;

                        ++point;
                        ++tags;

                        vec1 = *--point;
                        vec2 = *--point;

                        if (point <= limit) {
                            FT_Vector vec;
                            vec.x = point->x;
                            vec.y = point->y;

                            FT_Stroker_CubicTo(stroker, &vec1, &vec2, &vec);
                        } else {
                            FT_Stroker_CubicTo(stroker, &vec1, &vec2, v_start);
                        }
                    }
                }
            }

        Close:
            if (point > limit) return -1;

            if (!stroker->first_point)
                FT_Stroker_EndSubPath(stroker);

            n++;
        }

    }
}

namespace {

mlir::LogicalResult myExponentialFunction(mlir::OpBuilder& rewriter, mlir::OperationState& opState) {
    auto expOp = dyn_cast_or_null<myCustomExpOp>(opState.getOperation());
    if (!expOp)
        return failure();

    auto inputType = expOp.getType();
    SmallVector<int64_t, 4> shape;
    for (auto dim : inputType.cast<mlir::RankedTensorType>().getShape())
        shape.push_back(dim);

    auto inputOperand = opState.operands()[0];
    auto inputValue = rewriter.create<myCustomLoadOp>(opState.location, inputOperand.getType(), inputOperand);

    mlir::Value n = rewriter.create<arith::ConstantIndexOp>(opState.location, 127).getResult();
    auto i32VecType = rewriter.getIntegerType(32);
    auto broadcastN = rewriter.create<myCustomBroadcastOp>(opState.location, i32VecType, inputValue);

    mlir::Value nClamped = rewriter.create<arith::SelectOp>(
        opState.location,
        rewriter.getICmpEq(inputValue, n).getResults()[0],
        broadcastN,
        rewriter.create<arith::ConstantIndexOp>(opState.location, 0)
    );

    auto expC1 = rewriter.create<arith::ConstantIndexOp>(opState.location, -2.875);
    auto expC2 = rewriter.create<arith::ConstantIndexOp>(opState.location, -3.64);

    mlir::Value xUpdated = rewriter.create<myCustomFMulAddOp>(
        opState.location,
        inputValue,
        expC1.getResult(),
        nClamped
    );

    xUpdated = rewriter.create<myCustomFMulAddOp>(
        opState.location,
        xUpdated,
        expC2.getResult(),
        nClamped
    );

    mlir::Value zPoly0 = rewriter.create<arith::ConstantIndexOp>(opState.location, 1.67);
    mlir::Value zPoly1 = rewriter.create<arith::ConstantIndexOp>(opState.location, -2.835);
    mlir::Value zPoly2 = rewriter.create<arith::ConstantIndexOp>(opState.location, 0.6945);
    mlir::Value zPoly3 = rewriter.create<arith::ConstantIndexOp>(opState.location, -0.1275);
    mlir::Value zPoly4 = rewriter.create<arith::ConstantIndexOp>(opState.location, 0.00869);

    auto mulXSquare = rewriter.create<myCustomFMulOp>(opState.location, xUpdated, xUpdated);

    mlir::Value z = rewriter.create<myCustomFMulAddOp>(
        opState.location,
        zPoly0.getResult(),
        xUpdated,
        zPoly1
    );

    z = rewriter.create<myCustomFMulAddOp>(
        opState.location,
        z,
        xUpdated,
        zPoly2
    );

    z = rewriter.create<myCustomFMulAddOp>(
        opState.location,
        z,
        xUpdated,
        zPoly3
    );

    z = rewriter.create<myCustomFMulAddOp>(
        opState.location,
        z,
        mulXSquare.getResult(),
        xUpdated
    );

    z = rewriter.create<myCustomFMulAddOp>(
        opState.location,
        z,
        xUpdated,
        zPoly4
    );

    auto oneConst = rewriter.create<arith::ConstantIndexOp>(opState.location, 1);
    z = rewriter.create<myCustomFMulAddOp>(
        opState.location,
        oneConst.getResult(),
        mulXSquare.getResult(),
        z
    );

    auto exp2I32Op = rewriter.create<myCustomExp2Op>(opState.location, i32VecType, nClamped);

    mlir::Value ret = rewriter.create<myCustomFMulOp>(
        opState.location,
        z,
        exp2I32Op
    );

    rewriter.replaceOp(opState.getOperation(), {ret});
    return success();
}

} // namespace

// Inner Local Optimization Ransac.
        for (int iter = 0; iter < lo_inner_max_iterations; iter++) {
            int num_estimated_models;
            // Generate sample of lo_sample_size from inliers from the best model.
            if (num_inlier_of_best_model > lo_sample_size) {
                // if there are many inliers take limited number at random.
                num_estimated_models = estimator->estimateModelNonMinimalSample
                        (lo_sampler->generateUniqueRandomSubset(inlier_of_best_model,
                                num_inlier_of_best_model), lo_sample_size, lo_models, weights);
            } else {
                // if model was not updated in first iteration, so break.
                if (iter > 0) break;
                // if inliers are less than limited number of sample then take all for estimation
                // if it fails -> end Lo.
                num_estimated_models = estimator->estimateModelNonMinimalSample
                    (inlier_of_best_model, num_inlier_of_best_model, lo_models, weights);
            }

            //////// Choose the best lo_model from estimated lo_models.
            for (int model_idx = 0; model_idx < num_estimated_models; model_idx++) {
                const Score temp_score = quality->getScore(lo_model[model_idx]);
                if (temp_score.isBetter(new_model_score)) {
                    new_model_score = temp_score;
                    lo_model[model_idx].copyTo(new_model);
                }
            }

            if (is_iterative) {
                double lo_threshold = new_threshold;
                // get max virtual inliers. Note that they are nor real inliers,
                // because we got them with bigger threshold.
                int virtual_inlier_size = quality->getInliers
                        (new_model, virtual_inliers, lo_threshold);

                Mat lo_iter_model;
                Score lo_iter_score = Score(); // set worst case
                for (int iterations = 0; iterations < lo_iter_max_iterations; iterations++) {
                    lo_threshold -= threshold_step;

                    if (virtual_inlier_size > lo_iter_sample_size) {
                        // if there are more inliers than limit for sample size then generate at random
                        // sample from LO model.
                        num_estimated_models = estimator->estimateModelNonMinimalSample
                                (lo_iter_sampler->generateUniqueRandomSubset (virtual_inliers,
                            virtual_inlier_size), lo_iter_sample_size, lo_iter_models, weights);
                    } else {
                        // break if failed, very low probability that it will not fail in next iterations
                        // estimate model with all virtual inliers
                        num_estimated_models = estimator->estimateModelNonMinimalSample
                                (virtual_inliers, virtual_inlier_size, lo_iter_models, weights);
                    }
                    if (num_estimated_models == 0) break;

                    // Get score and update virtual inliers with current threshold
                    ////// Choose the best lo_iter_model from estimated lo_iter_models.
                    lo_iter_models[0].copyTo(lo_iter_model);
                    lo_iter_score = quality->getScore(lo_iter_model);
                    for (int model_idx = 1; model_idx < num_estimated_models; model_idx++) {
                        const Score temp_score = quality->getScore(lo_iter_models[model_idx]);
                        if (temp_score.isBetter(lo_iter_score)) {
                            lo_iter_score = temp_score;
                            lo_iter_models[model_idx].copyTo(lo_iter_model);
                        }
                    }

                    if (iterations != lo_iter_max_iterations-1)
                        virtual_inlier_size = quality->getInliers(lo_iter_model, virtual_inliers, lo_threshold);
                }

                if (lo_iter_score.isBetter(new_model_score)) {
                    new_model_score = lo_iter_score;
                    lo_iter_model.copyTo(new_model);
                }
            }

            if (num_inlier_of_best_model < new_model_score.inlier_number && iter != lo_inner_max_iterations-1)
                num_inlier_of_best_model = quality->getInliers (new_model, inlier_of_best_model);
        }

