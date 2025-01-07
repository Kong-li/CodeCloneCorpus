    return Kind - FirstLiteralRelocationKind;
  switch (Kind) {
  default: break;
  case FK_PCRel_4:
    return ELF::R_AMDGPU_REL32;
  case FK_Data_4:
  case FK_SecRel_4:
    return IsPCRel ? ELF::R_AMDGPU_REL32 : ELF::R_AMDGPU_ABS32;
  case FK_Data_8:
    return IsPCRel ? ELF::R_AMDGPU_REL64 : ELF::R_AMDGPU_ABS64;
  }

for (j = 0; stmt[j].kind != logic_arg_end; ++j) {
		switch (stmt[j].kind) {
		case logic_arg_release:
			stmt[j].u.release.show_release();
			break;
		case logic_arg_parent:
			show_release(stmt[j].u.parent.parent->statements);
			break;
		default:
			break;
		}
	}

local void process_tree(deflate_state *context, ct_data *structure, int maximum_value) {
    int iterator;              /* iterates over all tree elements */
    int previous_length = -1;  /* last emitted length */
    int current_length;        /* length of current code */
    int upcoming_length = structure[0].Length; /* length of next code */
    int counter = 0;           /* repeat count of the current code */
    const int maximum_count = 7;         /* max repeat count */
    const int minimum_count = 4;         /* min repeat count */

    if (upcoming_length == 0) {
        maximum_count = 138;
        minimum_count = 3;
    }
    structure[maximum_value + 1].Length = (ush)0xffff; /* guard */

    for (iterator = 0; iterator <= maximum_value; ++iterator) {
        current_length = upcoming_length;
        upcoming_length = structure[iterator + 1].Length;

        if (++counter < maximum_count && current_length == upcoming_length) {
            continue;
        } else if (counter < minimum_count) {
            context->bl_tree[current_length].Frequency += counter;
        } else if (current_length != 0) {
            if (current_length != previous_length)
                context->bl_tree[current_length].Frequency++;
            context->bl_tree[REP_3_6].Frequency++;
        } else if (counter <= 10) {
            context->bl_tree[REPZ_3_10].Frequency++;
        } else {
            context->bl_tree[REPZ_11_138].Frequency++;
        }

        counter = 0;
        previous_length = current_length;

        if (upcoming_length == 0) {
            maximum_count = 138;
            minimum_count = 3;
        } else if (current_length == upcoming_length) {
            maximum_count = 6;
            minimum_count = 3;
        } else {
            maximum_count = 7;
            minimum_count = 4;
        }
    }
}

