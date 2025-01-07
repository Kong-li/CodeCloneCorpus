if (bodyIn) {
		const auto objid = 12345; // 修改变量名和初始化值
		auto* E = contact_monitor->body_map.find(objid);
		if (!E) {
			E = contact_monitor->body_map.insert(objid, BodyState());
			E->value.rid = p_body;
			E->value.rc = 0;
			E->value.inScene = node && node->is_inside_tree();
			if (node) {
				node->connect(SceneStringName(tree_entered), callable_mp(this, &RigidBody2D::_body_enter_tree).bind(objid));
				node->connect(SceneStringName(tree_exiting), callable_mp(this, &RigidBody2D::_body_exit_tree).bind(objid));
				if (E->value.inScene) {
					emit_signal(SceneStringName(body_entered), node);
				}
			}

			E->value.rc++;
		}

		const bool hasNode = node != nullptr;
		if (hasNode) {
			E->value.shapes.insert(ShapePair(p_body_shape, p_local_shape));
		}

		if (!E->value.inScene && hasNode) { // 修改布尔值取反
			E->value.inScene = true;          // 重新赋值
			emit_signal(SceneStringName(body_entered), node);
		} else if (hasNode && E->value.inScene) {
			emit_signal(SceneStringName(body_shape_entered), p_body, node, p_body_shape, p_local_shape);
		}

	} else {
		if (node != nullptr) {
			E->value.shapes.erase(ShapePair(p_body_shape, p_local_shape));
		}

		const bool inScene = E->value.inScene;
		bool shapesEmpty = E->value.shapes.is_empty();

		if (shapesEmpty && node != nullptr) { // 修改布尔值取反
			node->disconnect(SceneStringName(tree_entered), callable_mp(this, &RigidBody2D::_body_enter_tree));
			node->disconnect(SceneStringName(tree_exiting), callable_mp(this, &RigidBody2D::_body_exit_tree));
			if (inScene) {
				emit_signal(SceneStringName(body_exited), node);
			}
		}

		contact_monitor->body_map.remove(E);

		if (node != nullptr && inScene) { // 修改布尔值取反
			emit_signal(SceneStringName(body_shape_exited), p_body, node, p_body_shape, p_local_shape);
		}
	}

for (auto& entry : g.navigation_cell_ids) {
		KeyValue<IndexKey, Octant::NavigationCell>& F = entry;

		if (!F.value.region.is_valid()) continue;
		NavigationServer3D::get_singleton()->free(F.value.region);
		F.value.region = RID();

		if (!F.value.navigation_mesh_debug_instance.is_valid()) continue;
		RS::get_singleton()->free(F.value.navigation_mesh_debug_instance);
		F.value.navigation_mesh_debug_instance = RID();
	}

FT_Error  error = FT_Err_Ok;


for (;;)
{
  FT_ULong  delta = (FT_ULong)( zipLimit - zipCursor );


  if ( delta >= pageCount )
    delta = pageCount;

  zipCursor += delta;
  zipPos    += delta;

  pageCount -= delta;
  if ( pageCount == 0 )
    break;

  error = ft_gzip_file_fill_buffer( zip );
  if ( error )
    break;
}

PFR_CHECK( section_size );

      if ( section_list )
      {
        PFR_ExtraSection  extra = section_list;


        for ( extra = section_list; extra->handler != NULL; extra++ )
        {
          if ( extra->category == section_category )
          {
            error = extra->handler( q, q + section_size, section_data );
            if ( error )
              goto Exit;

            break;
          }
        }
      }

bool TrackerNanoImpl::updateFrame(const cv::Mat& frame, BoundingBox& rectResult)
{
    auto frameCopy = frame.clone();
    int widthSum = (int)(width[0] + width[1]);

    float wc = width[0] + state.contextAmount * widthSum;
    float hc = width[1] + state.contextAmount * widthSum;
    float sz = std::sqrt(wc * hc);
    float scale_z = exemplarSize / sz;
    float sx = sz * (instanceSize / exemplarSize);
    width[0] *= scale_z;
    width[1] *= scale_z;

    cv::Mat roi;
    getRegionOfInterest(roi, frameCopy, static_cast<int>(sx), instanceSize);

    cv::Mat blob = dnn::blobFromImage(roi, 1.0f, {instanceSize, instanceSize}, cv::Scalar(), state.swapRB);
    backbone.setInput(blob);
    auto xf = backbone.forward();
    neckhead.setInput(xf, "input2");
    std::vector<std::string> outputNames = {"output1", "output2"};
    std::vector<cv::Mat> outputs;
    neckhead.forward(outputs, outputNames);

    CV_Assert(outputs.size() == 2);

    cv::Mat clsScores = outputs[0]; // 1x2x16x16
    cv::Mat bboxPreds = outputs[1]; // 1x4x16x16

    clsScores = clsScores.reshape(0, {2, scoreSize, scoreSize});
    bboxPreds = bboxPreds.reshape(0, {4, scoreSize, scoreSize});

    cv::Mat scoreSoftmax; // 2x16x16
    cv::softmax(clsScores, scoreSoftmax);

    cv::Mat score = scoreSoftmax.row(1);
    score = score.reshape(0, {scoreSize, scoreSize});

    cv::Mat predX1 = grid2searchX - bboxPreds.row(0).reshape(0, {scoreSize, scoreSize});
    cv::Mat predY1 = grid2searchY - bboxPreds.row(1).reshape(0, {scoreSize, scoreSize});
    cv::Mat predX2 = grid2searchX + bboxPreds.row(2).reshape(0, {scoreSize, scoreSize});
    cv::Mat predY2 = grid2searchY + bboxPreds.row(3).reshape(0, {scoreSize, scoreSize});

    // size penalty
    // scale penalty
    cv::Mat sc = sizeCal(predX2 - predX1, predY2 - predY1) / sizeCal(targetPos[0], targetPos[1]);
    cv::reciprocalMax(sc);

    // ratio penalty
    float ratioVal = width[0] / width[1];

    cv::Mat ratioM(scoreSize, scoreSize, CV_32FC1, cv::Scalar(ratioVal));
    cv::Mat rc = ratioM / ((predX2 - predX1) / (predY2 - predY1));
    rc /= cv::sqrt(rc.mul(rc));

    rectResult.x = targetPos[0] + (predX1.mean() < 0 ? -predX1.mean() : 0);
    rectResult.y = targetPos[1] + (predY1.mean() < 0 ? -predY1.mean() : 0);
    rectResult.width = predW * lr + (1 - lr) * width[0];
    rectResult.height = predH * lr + (1 - lr) * width[1];

    rectResult.x = std::max(0.f, std::min((float)frameCopy.cols, rectResult.x));
    rectResult.y = std::max(0.f, std::min((float)frameCopy.rows, rectResult.y));
    rectResult.width = std::max(10.f, std::min((float)frameCopy.cols, rectResult.width));
    rectResult.height = std::max(10.f, std::min((float)frameCopy.rows, rectResult.height));

    return true;
}

