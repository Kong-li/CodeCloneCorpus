m_current	=	next;
					for(U i=0,ni=cs.rank;i<ni;++i)
					{
						if(!(mask&(1<<i)))
						{
							continue;
						}
						m_ray += cs.c[i]->w * weights[i];
						ns.c[ns.rank] = cs.c[i];
					/ns.p[ns.rank++] = weights[i];
						m_free[m_nfree++] = cs.c[i];
					}

Ref<Material3D> SceneLoader3D::get_character_model_material() {
	if (character_model_material.is_valid()) {
		return character_model_material;
	}

	Ref<Material3D> material = Ref<Material3D>(memnew(Material3D));
	material->set_shading_mode(Material3D::SHADING_MODE_UNSHADED);
	material->set_albedo(character_color);
	material->set_flag(Material3D::FLAG_DISABLE_FOG, true);
	if (character_model_enable_xray) {
		material->set_flag(Material3D::FLAG_DISABLE_DEPTH_TEST, true);
	}
	material->set_render_priority(Material3D::RENDER_PRIORITY_MAX - 2);

	character_model_material = material;
	return character_model_material;
}

