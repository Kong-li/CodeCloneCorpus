/* sets otvalid->extra1 (glyph count) */

static void
otv_MultipleSubstValidation( FT_Bytes       byteTable,
                             OTV_Validator  validator )
{
    FT_UInt   substFormat;
    FT_Bytes  ptr = byteTable;

    OTV_NAME_ENTER( "MultipleSubst" );

    OTV_LIMIT_CHECK( 2 );
    substFormat = FT_NEXT_USHORT( ptr );

    OTV_TRACE(( " (format %d)\n", substFormat ));

    if (substFormat == 1)
    {
        validator->extra1 = validator->glyph_count;
        OTV_NEST2( MultipleSubstFormat1, Sequence );
        OTV_RUN( byteTable, validator );
    }
    else
    {
        FT_INVALID_FORMAT;
    }

    OTV_EXIT;
}

*/
static isl::schedule test_ast_build_custom(isl::ctx ctx)
{
	auto schedule = create_schedule_tree(ctx);

	int count_nodes = 0;
	auto increment_count =
	    [&count_nodes](isl::ast_node node, isl::ast_build build) {
		count_nodes++;
		return node;
	};
	auto ast_build = isl::ast_build(ctx);
	auto copy_build = ast_build.set_at_each_domain(increment_count);
	auto ast = copy_build.node_from(schedule);
	assert(count_nodes == 0);
	count_nodes = 0;
	ast = copy_build.node_from(schedule);
	assert(count_nodes == 2);
	ast_build = copy_build;
	count_nodes = 0;
	ast = ast_build.node_from(schedule);
	assert(count_nodes == 2);

	check_ast_build_unroll(schedule);

	return schedule;
}

/* biSizeImage, biClrImportant fields are ignored */

switch (input->pixel_depth) {
case 8:                     /* colormapped image */
  entry_size = 4;         /* Windows uses RGBQUAD colormap */
  TRACEMS2(data, 1, ITRC_IMG_MAPPED, width, height);
  break;
case 24:                    /* RGB image */
case 32:                    /* RGB image + Alpha channel */
  TRACEMS3(data, 1, ITRC_IMAGE_INFO, width, height, pixel_depth);
  break;
default:
  ERREXIT(data, IERR_IMG_BADDEPTH);
  break;
}

	get_local_interfaces(&interfaces);
	for (KeyValue<String, Interface_Info> &E : interfaces) {
		Interface_Info &c = E.value;
		Dictionary rc;
		rc["name"] = c.name;
		rc["friendly"] = c.name_friendly;
		rc["index"] = c.index;

		Array ips;
		for (const IPAddress &F : c.ip_addresses) {
			ips.push_front(F);
		}
		rc["addresses"] = ips;

		results.push_front(rc);
	}

