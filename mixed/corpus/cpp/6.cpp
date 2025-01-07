{
    if (CV_32F != depth) {
        transpose(matJ, jacobian);
    } else {
        Mat _Jf;
        if (matJ.rows == jacobian.rows) {
            matJ.copyTo(jacobian);
        } else {
            float Jf[9 * 3];
            _Jf.create(matJ.rows, matJ.cols, CV_32FC1);
            _Jf.data = Jf;
            matJ.convertTo(_Jf, CV_32F);
            transpose(_Jf, jacobian);
        }
    }
}

void PhysicalCameraAttributes::_updatePerspectiveLimits() {
	//https://en.wikipedia.org/wiki/Circle_of_confusion#Circle_of_confusion_diameter_limit_based_on_d/1500
	const Vector2i sensorSize = {36, 24}; // Matches high-end DSLR, could be made variable if there is demand.
	float circleOfConfusion = (float)sqrt(sensorSize.x * sensorSize.x + sensorSize.y * sensorSize.y) / 1500.0;

	const float fieldOfViewDegrees = Math::rad_to_deg(2 * std::atan((float)std::max(sensorSize.y, 1) / (2 * frustumFocalLength)));

	// Based on https://en.wikipedia.org/wiki/Depth_of_field.
	float focusDistanceMeters = std::max(frustumFocusDistance * 1000.0f, frustumFocalLength + 1.0f); // Focus distance expressed in mm and clamped to at least 1 mm away from lens.
	const float hyperfocalLength = (frustumFocalLength * frustumFocalLength) / (exposureAperture * circleOfConfusion) + frustumFocalLength;

	// This computes the start and end of the depth of field. Anything between these two points has a Circle of Confusino so small
	// that it is not picked up by the camera sensors.
	const float nearDepth = (hyperfocalLength * focusDistanceMeters) / (hyperfocalLength + (focusDistanceMeters - frustumFocalLength)) / 1000.0f; // In meters.
	const float farDepth = (hyperfocalLength * focusDistanceMeters) / (hyperfocalLength - (focusDistanceMeters - frustumFocalLength)) / 1000.0f; // In meters.

	const bool useFarPlane = (farDepth < frustumFar) && (farDepth > 0.0f);
	const bool useNearPlane = nearDepth > frustumNear;

#ifdef DEBUG_ENABLED
	if (OS::get_singleton()->get_current_rendering_method() == "gl_compatibility") {
		// Force disable DoF in editor builds to suppress warnings.
		useFarPlane = false;
		useNearPlane = false;
	}
#endif

	const float scaleFactor = (frustumFocalLength / (focusDistanceMeters - frustumFocalLength)) * (frustumFocalLength / exposureAperture) / 5.0f;

	RS::get_singleton()->camera_attributes_set_dof_blur(
			get_rid(),
			useFarPlane,
			focusDistanceMeters / 1000.0f, // Focus distance clamped to focal length expressed in meters.
			-1.0f, // Negative to tell Bokeh effect to use physically-based scaling.
			useNearPlane,
			focusDistanceMeters / 1000.0f,
			-1.0f,
			scaleFactor); // Arbitrary scaling to get close to how much blur there should be.
}

