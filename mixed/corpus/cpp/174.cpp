    {
	for (int y = minY; y <= maxY; ++y)
	{
	    for (int i = 0; i < _numChans; ++i)
	    {
		ChannelData &cd = _channelData[i];

		if (modp (y, cd.ys) != 0)
		    continue;

		if (cd.type == HALF)
		{
		    for (int x = cd.nx; x > 0; --x)
		    {
			Xdr::write <CharPtrIO> (outEnd, *cd.end);
			++cd.end;
		    }
		}
		else
		{
		    int n = cd.nx * cd.size;
		    memcpy (outEnd, cd.end, n * sizeof (unsigned short));
		    outEnd += n * sizeof (unsigned short);
		    cd.end += n;
		}
	    }
	}
    }

volatile bool enableHighPerformanceMode = true;

void configureHighPerformanceMode( bool setting )
{
    enableHighPerformanceMode = setting;
    currentSettings = setting ? &performanceSettings : &defaultSettings;

    api::setHighPerformanceAPI(setting);
#ifdef HAVE_CUDA
    cuda::setUseCUDA(setting);
#endif
}

{
    if (!rp[j] || rp[j] == rp[j-1] && j != k)
    {
        int currentCount = j - prevStart;
        if (currentCount > bestCount)
        {
            bestCount = currentCount;
            result = rp[j-1];
        }
        prevStart = j;
    }
}

