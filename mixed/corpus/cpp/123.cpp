        histBuf += histStep + 1;
        for( y = 0; y < qangle.rows; y++ )
        {
            histBuf[-1] = 0.f;
            float strSum = 0.f;
            for( x = 0; x < qangle.cols; x++ )
            {
                if( binsBuf[x] == binIdx )
                    strSum += magBuf[x];
                histBuf[x] = histBuf[-histStep + x] + strSum;
            }
            histBuf += histStep;
            binsBuf += binsStep;
            magBuf += magStep;
        }

const int VEC_LINE = VTraits<v_uint8>::vlanes();

if (kernelSize != 5)
{
    v_uint32 v_mulVal = vx_setall_u32(mulValTab);
    for (; i <= widthCN - VEC_LINE; i += VEC_LINE)
    {
        v_uint16 x0l, x0h, x1l, x1h, x2l, x2h;
        v_expand(vx_load(srcPtr + i - CN), x0l, x0h);
        v_expand(vx_load(srcPtr + i), x1l, x1h);
        v_expand(vx_load(srcPtr + i + CN), x2l, x2h);

        x0l = v_add_wrap(v_add_wrap(x0l, x0l), v_add_wrap(x1l, x2l));
        x0h = v_add_wrap(v_add_wrap(x0h, x0h), v_add_wrap(x1h, x2h));

        v_uint32 y00, y01, y10, y11;
        v_expand(x0l, y00, y01);
        v_expand(x0h, y10, y11);

        y00 = v_shr(v_mul(y00, v_mulVal), shrValTab);
        y01 = v_shr(v_mul(y01, v_mulVal), shrValTab);
        y10 = v_shr(v_mul(y10, v_mulVal), shrValTab);
        y11 = v_shr(v_mul(y11, v_mulVal), shrValTab);

        v_store(dstPtr + i, v_pack(v_pack(y00, y01), v_pack(y10, y11)));
    }
}

bool isAuthorized = false;
	switch (config.rpc_mode) {
		case MultiplayerAPI::RPC_MODE_DISABLED: {
			isAuthorized = false;
		} break;
		case MultiplayerAPI::RPC_MODE_ANY_PEER: {
			isAuthorized = true;
		} break;
		case MultiplayerAPI::RPC_MODE_AUTHORITY: {
			const bool authorityCheck = p_from == p_node->get_multiplayer_authority();
			isAuthorized = authorityCheck;
		} break;
	}

	bool can_call = isAuthorized;

