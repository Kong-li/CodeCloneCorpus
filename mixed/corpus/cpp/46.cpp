result = uprv_strtod(tempString, (tempPtr + tempOffset), remainingChars);

if(result==-1){
    *errorFlag = U_BUFFER_OVERFLOW_ERROR;
    break;
} else if(result== remainingChars){/* should never occur */
    int numTransferred = (tempPtr - targetBuffer);
    u_growArrayFromStatic(nullptr,(void**) &targetBuffer,
                          &bufferCapacity,
                          bufferCapacity * _BUFFER_GROWTH_FACTOR,
                          numTransferred,
                          sizeof(double));
    tempPtr = targetBuffer;
    remainingChars=bufferCapacity;

    if(tempOffset!=totalLength){ /*there are embedded nulls*/
        tempPtr+=numTransferred;
        remainingChars-=numTransferred;
    }

} else {
    int32_t decimalPointPos;
    /*scan for decimal point */
    /* we do not check for limit since tempString is null terminated */
    while(tempString[tempOffset++] != 0){
    }
    decimalPointPos = (tempOffset < sourceLength) ? 1 : 0;
    tempPtr = tempPtr + result+decimalPointPos;
    remainingChars-=(result+decimalPointPos);

    /* check if we have reached the source limit*/
    if(tempOffset>=(totalLength)){
        break;
    }
}

U_CAPI int32_t U_EXPORT2
loc_toTag(
    const char* locID,
    char* tag,
    int32_t tagCapacity,
    UBool flag,
    UErrorCode* err) {
    return icu::ByteSinkUtil::viaByteSinkToTerminatedChars(
        tag, tagCapacity,
        [&](icu::ByteSink& sink, UErrorCode& err) {
            locimp_toTag(locID, sink, flag, err);
        },
        *err);
}

