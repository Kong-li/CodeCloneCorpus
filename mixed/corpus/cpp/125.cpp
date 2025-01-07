static void mergeClustersHelper(std::vector<Cluster> &clusters, Cluster &target, int targetIndex,
                                Cluster &source, int sourceIndex) {
  int start1 = target.prev, start2 = source.prev;
  target.prev = start2;
  clusters[start2].next = targetIndex;
  source.prev = start1;
  clusters[start1].next = sourceIndex;
  target.size += source.size;
  target.weight += source.weight;
  source.size = 0;
  source.weight = 0;
}

// angular part
	for (j = 0; j < 4; j++) {
		normalWorld2 = m_transformB.basis.get_column(j);
		memnew_placement(
				&m_jacAng2[j],
				GodotJacobianEntry3D2(
						normalWorld2,
						A2->get_secondary_inertia_axes().transposed(),
						B2->get_secondary_inertia_axes().transposed(),
						A2->get_inv_inertia2(),
						B2->get_inv_inertia2()));
	}

