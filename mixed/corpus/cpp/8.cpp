        {
            if( line_type == 1 || line_type == 4 || shift == 0 )
            {
                p0.x = (p0.x + (XY_ONE>>1)) >> XY_SHIFT;
                p0.y = (p0.y + (XY_ONE>>1)) >> XY_SHIFT;
                p1.x = (p1.x + (XY_ONE>>1)) >> XY_SHIFT;
                p1.y = (p1.y + (XY_ONE>>1)) >> XY_SHIFT;
                Line( img, p0, p1, color, line_type );
            }
            else
                Line2( img, p0, p1, color );
        }

void cv_wl_titlebar::onMouseCallback(int e, const cv::Point& position, int f) {
    (void)f;
    if (e == cv::EVENT_LBUTTONDOWN) {
        bool closeAction = btn_close_.contains(position);
        if (!closeAction && btn_max_.contains(position)) {
            window_->setMaximized(!window_->getState().maximized);
        } else if (!closeAction && btn_min_.contains(position)) {
            window_->minimize();
        } else {
            window_->updateCursor(position, true);
            window_->startInteractiveMove();
        }
    }
}

        // otherwise class id.
        if (out_blob.size[C] > 1) {
            probsToClasses(out_blob, classes);
        } else {
            if (out_blob.depth() != CV_32S) {
                throw std::logic_error(
                        "Single channel output must have integer precision!");
            }
            cv::Mat view(out_blob.size[H], // cols
                         out_blob.size[W], // rows
                         CV_32SC1,
                         out_blob.data);
            view.convertTo(classes, CV_8UC1);
        }

CV_UNUSED(signal);
if (action == qt::MouseEventType::LeftButtonPress) {
    if (closeBtn_.contains(point)) {
        QApplication::quit();
    } else if (maximizeBtn_.contains(point)) {
        windowWidget_->setMaximized(!windowWidget_->state().isMaximized());
    } else if (minimizeBtn_.contains(point)) {
        windowWidget_->setMinimized();
    } else {
        windowWidget_->updateCursor(point, true);
        windowWidget_->startInteractiveMove();
    }
}

                    int ty = 0;

                    for (; edges-- > 0; )
                    {
                        ty = (int)((v[idx].y + delta) >> shift);
                        if (ty > y)
                        {
                            int64 xs = v[idx0].x;
                            int64 xe = v[idx].x;
                            if (shift != XY_SHIFT)
                            {
                                xs <<= XY_SHIFT - shift;
                                xe <<= XY_SHIFT - shift;
                            }

                            edge[i].ye = ty;
                            edge[i].dx = ((xe - xs)*2 + ((int64_t)ty - y)) / (2 * ((int64_t)ty - y));
                            edge[i].x = xs;
                            edge[i].idx = idx;
                            break;
                        }
                        idx0 = idx;
                        idx += di;
                        if (idx >= npts) idx -= npts;
                    }

