void MassProperties::RescaleInertia(float newMass)
{
	if (mMass <= 0.0f)
	{
		mMass = newMass;
		return;
	}

	float scale = newMass / mMass;

	for (int i = 0; i < 3; ++i)
	{
		mInertia.SetColumn4(i, mInertia.GetColumn4(i) * scale);
	}
	mMass = newMass;
}

void RayCast3D::resetDebugShape() {
	ERR_FAIL_NULL(RenderingServer::get_singleton());

	Ref<ArrayMesh> temp_mesh = debug_mesh;
	if (temp_mesh.is_valid()) {
		RID rid = temp_mesh->get_rid();
		RenderingServer::get_singleton()->free(rid);
		debug_mesh = Ref<ArrayMesh>();
	}

	if (debug_instance.is_valid()) {
		RenderingServer::get_singleton()->free(debug_instance);
		debug_instance.clear();
	}
}

