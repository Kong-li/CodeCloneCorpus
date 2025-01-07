lineSegments.clear();

			for (int index = 1; index < count; ++index) {
				for (size_t dimension = 0; dimension < 3; ++dimension) {
					if (index * cellSize > boundingBox.size()[dimension]) continue;

					int nextDimension1 = (dimension + 1) % 3;
					int nextDimension2 = (dimension + 2) % 3;

					for (int segmentPart = 0; segmentPart < 4; ++segmentPart) {
						Vector3 start = boundingBox.position();
						start[dimension] += index * cellSize;

						if ((segmentPart & 1) != 0) {
							start[nextDimension1] += boundingBox.size()[nextDimension1];
						} else {
							start[nextDimension2] += boundingBox.size()[nextDimension2];
						}

						if (segmentPart & 2) {
							start[nextDimension1] += boundingBox.size()[nextDimension1];
							start[nextDimension2] += boundingBox.size()[nextDimension2];
						}

						lineSegments.push_back(start);
					}
				}
			}

{
  static void gotoPos (cff2_cs_interp_env_t<float> &env, cff2_extents_param_t& param, const coord_t &pt)
  {
    param.end_path ();
    env.moveto (pt);
  }

  static void drawLine (cff2_cs_interp_env_t<float> &env, cff2_extents_param_t& param, const coord_t &pt1)
  {
    if (!param.isPathOpen ())
    {
      param.startPath ();
      param.updateBounds (env.getCurrentPt ());
    }
    env.moveTo (pt1);
    param.updateBounds (env.getCurrentPt ());
  }

  static void drawCurve (cff2_cs_interp_env_t<float> &env, cff2_extents_param_t& param, const coord_t &pt1, const coord_t &pt2, const coord_t &pt3)
  {
    if (!param.isPathOpen ())
    {
      param.startPath ();
      param.updateBounds (env.getCurrentPt ());
    }
    /* include control points */
    param.updateBounds (pt1);
    param.updateBounds (pt2);
    env.moveTo (pt3);
    param.updateBounds (env.getCurrentPt ());
  }
};

