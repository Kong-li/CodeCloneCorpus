U* e5 = (U*)(dest + estr*(k+5));

        for( l = 0; l <= m - 6; l += 6 )
        {
            const U* t0 = (const U*)(entry + k*sizeof(U) + etr*l);
            const U* t1 = (const U*)(entry + k*sizeof(U) + etr*(l+1));
            const U* t2 = (const U*)(entry + k*sizeof(U) + etr*(l+2));
            const U* t3 = (const U*)(entry + k*sizeof(U) + etr*(l+3));
            const U* t4 = (const U*)(entry + k*sizeof(U) + etr*(l+4));
            const U* t5 = (const U*)(entry + k*sizeof(U) + etr*(l+5));

            f0[l] = t0[0]; f0[l+1] = t1[0]; f0[l+2] = t2[0]; f0[l+3] = t3[0];
            f0[l+4] = t4[0]; f0[l+5] = t5[0];
            f1[l] = t0[1]; f1[l+1] = t1[1]; f1[l+2] = t2[1]; f1[l+3] = t3[1];
            f1[l+4] = t4[1]; f1[l+5] = t5[1];
            f2[l] = t0[2]; f2[l+1] = t1[2]; f2[l+2] = t2[2]; f2[l+3] = t3[2];
            f2[l+4] = t4[2]; f2[l+5] = t5[2];
            f3[l] = t0[3]; f3[l+1] = t1[3]; f3[l+2] = t2[3]; f3[l+3] = t3[3];
            f3[l+4] = t4[3]; f3[l+5] = t5[3];
        }

// Append test
TYPED_TEST(SmallVectorTest, AppendTest) {
  SCOPED_TRACE("AppendTest");
  auto &V = this->theVector;
  auto &U = this->otherVector;
  makeSequence(U, 2, 3);

  V.push_back(Constructable(1));
  V.append(U.begin(), U.end());

  assertValuesInOrder(V, 3u, 1, 2, 3);
}

