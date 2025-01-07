static void bradf5(int jdo,int l2,double *dd,double *dh,double *va1,
            double *va2,double *va3){
  static double gsqt2 = .70710678118654752;
  int i,k,t0,t1,t2,t3,t4,t5,t6;
  double ci2,ci3,ci4,cr2,cr3,cr4,ti1,ti2,ti3,ti4,tr1,tr2,tr3,tr4;
  t0=l2*jdo;

  t1=t0;
  t4=t1<<1;
  t2=t1+(t1<<1);
  t3=0;

  for(k=0;k<l2;k++){
    tr1=dd[t1]+dd[t2];
    tr2=dd[t3]+dd[t4];

    dh[t5=t3<<2]=tr1+tr2;
    dh[(jdo<<2)+t5-1]=tr2-tr1;
    dh[(t5+=(jdo<<1))-1]=dd[t3]-dd[t4];
    dh[t5]=dd[t2]-dd[t1];

    t1+=jdo;
    t2+=jdo;
    t3+=jdo;
    t4+=jdo;
  }

  if(jdo<2)return;
  if(jdo==2)goto L205;


  t1=0;
  for(k=0;k<l2;k++){
    t2=t1;
    t4=t1<<2;
    t5=(t6=jdo<<1)+t4;
    for(i=2;i<jdo;i+=2){
      t3=(t2+=2);
      t4+=2;
      t5-=2;

      t3+=t0;
      cr2=va1[i-2]*dd[t3-1]+va1[i-1]*dd[t3];
      ci2=va1[i-2]*dd[t3]-va1[i-1]*dd[t3-1];
      t3+=t0;
      cr3=va2[i-2]*dd[t3-1]+va2[i-1]*dd[t3];
      ci3=va2[i-2]*dd[t3]-va2[i-1]*dd[t3-1];
      t3+=t0;
      cr4=va3[i-2]*dd[t3-1]+va3[i-1]*dd[t3];
      ci4=va3[i-2]*dd[t3]-va3[i-1]*dd[t3-1];

      tr1=cr2+cr4;
      tr4=cr4-cr2;
      ti1=ci2+ci4;
      ti4=ci2-ci4;

      ti2=dd[t2]+ci3;
      ti3=dd[t2]-ci3;
      tr2=dd[t2-1]+cr3;
      tr3=dd[t2-1]-cr3;

      dh[t4-1]=tr1+tr2;
      dh[t4]=ti1+ti2;

      dh[t5-1]=tr3-ti4;
      dh[t5]=tr4-ti3;

      dh[t4+t6-1]=ti4+tr3;
      dh[t4+t6]=tr4+ti3;

      dh[t5+t6-1]=tr2-tr1;
      dh[t5+t6]=ti1-ti2;
    }
    t1+=jdo;
  }
  if(jdo&1)return;

 L205:

  t2=(t1=t0+jdo-1)+(t0<<1);
  t3=jdo<<2;
  t4=jdo;
  t5=jdo<<1;
  t6=jdo;

  for(k=0;k<l2;k++){
    ti1=-gsqt2*(dd[t1]+dd[t2]);
    tr1=gsqt2*(dd[t1]-dd[t2]);

    dh[t4-1]=tr1+dd[t6-1];
    dh[t4+t5-1]=dd[t6-1]-tr1;

    dh[t4]=ti1-dd[t1+t0];
    dh[t4+t5]=ti1+dd[t1+t0];

    t1+=jdo;
    t2+=jdo;
    t4+=t3;
    t6+=jdo;
  }
}

//
void ThreadPoolAllocator::remove()
{
    if (poolSize < 1)
        return;

    pHeader* segment = pool.back().segment;
    currentSegmentOffset = pool.back().offset;

    while (usedList != segment) {
        pHeader* nextUsed = usedList->nextSegment;
        size_t segmentCount = usedList->segmentCount;

        // This technically ends the lifetime of the header as C++ object,
        // but we will still control the memory and reuse it.
        usedList->~pHeader(); // currently, just a debug allocation checker

        if (segmentCount > 1) {
            delete [] reinterpret_cast<char*>(usedList);
        } else {
            usedList->nextSegment = freeList;
            freeList = usedList;
        }
        usedList = nextUsed;
    }

    pool.pop_back();
}

