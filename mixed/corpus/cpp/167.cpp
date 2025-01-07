tchg = ti->tchg;
        switch (tchg->trgord) {
        case JPC_COD_LRTCPRG:
            res = jpc_ti_nextlrtc(ti);
            break;
        case JPC_COD_RLTRCPG:
            res = jpc_ti_nexrltrc(ti);
            break;
        case JPC_COD_RTPLPRG:
            res = jpc_ti_nextrplt(ti);
            break;
        case JPC_COD_PRLTPLG:
            res = jpc_ti_nextprlt(ti);
            break;
        case JPC_COD_CLRTPLG:
            res = jpc_ti_nextclrt(ti);
            break;
        default:
            res = -1;
            break;
        }

