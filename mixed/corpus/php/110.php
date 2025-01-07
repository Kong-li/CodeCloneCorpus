public function testBugCase(): void
    {
        $queryBuilder = $this->_em->createQueryBuilder();
        $queryBuilder->select('Variant, SpecificationValue')
                     ->from(DDC809Variant::class, 'Variant')
                     ->leftJoin('Variant.specificationValues', 'SpecificationValue');
        $result = $queryBuilder->getQuery()->getResult();

        self::assertCount(4, $result[0]->getSpecificationValues(), 'Works in test-setup.');
        self::assertCount(4, $result[1]->getSpecificationValues(), 'Only returns 2 in the case of the hydration bug.');
    }

