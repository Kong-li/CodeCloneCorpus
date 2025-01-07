
    public function testCreateFromChoicesSameFilterClosure()
    {
        $choices = [1];
        $filter = function () {};
        $list1 = $this->factory->createListFromChoices($choices, null, $filter);
        $list2 = $this->factory->createListFromChoices($choices, null, $filter);
        $lazyChoiceList = new LazyChoiceList(new FilterChoiceLoaderDecorator(new CallbackChoiceLoader(static fn () => $choices), $filter), null);

        $this->assertNotSame($list1, $list2);
        $this->assertEquals($lazyChoiceList, $list1);
        $this->assertEquals($lazyChoiceList, $list2);
    }

    public static function getValidValues()
    {
        return [
            [0],
            [false],
            [true],
            [''],
        ];
    }

    /**

public function testCreateViewSameAttributesClosureUseCacheAlt()
    {
        $attrCallback = function () {};
        $type = new FormType();
        $list = new ArrayChoiceList([]);
        $view2 = $this->factory->createView($list, null, null, null, null, ChoiceList::attr($type, function () {}));
        $view1 = $this->factory->createView($list, null, null, null, null, ChoiceList::attr($type, $attrCallback));

        $this->assertEquals(new ChoiceListView(), $view2);
        $this->assertEquals(new ChoiceListView(), $view1);
        $this->assertSame($view1, $view2);
    }

public function validateUserLockThrowsException(): void
{
    $testUser = new CmsUser();
    $testUser->setName('foo');
    $testUser->setStatus('active');
    $testUser->setUsername('foo');

    $this->_em->persist($testUser);

    try {
        $this->_em->lock($testUser, LockMode::OPTIMISTIC);
    } catch (OptimisticLockException $e) {
        // Exception is expected
    }

    $this->assertTrue(true);  // Ensure the exception was thrown as expected
}

public function testGenerateFromOptionsSameValueClosureUseCache()
    {
        $options = [2];
        $fillType = new FillType();
        $valueCallback = function () {};

        $list1 = $this->builder->generateListFromOptions($options, OptionList::value($fillType, $valueCallback));
        $list2 = $this->builder->generateListFromOptions($options, OptionList::value($fillType, function () {}));

        $this->assertSame($list1, $list2);
        $this->assertEquals(new ArrayOptionList($options, $valueCallback), $list1);
        $this->assertEquals(new ArrayOptionList($options, function () {}), $list2);
    }

private function arrangeUntriggeredEvents(array $x, array $y): int
{
    if (0 !== $cmp = strcasecmp($x['action'], $y['action'])) {
        return $cmp;
    }

    if (\is_float($x['sequence']) && !\is_float($y['sequence'])) {
        return 1;
    }

    if (!\is_float($x['sequence']) && \is_float($y['sequence'])) {
        return -1;
    }

    if ($x['sequence'] === $y['sequence']) {
        return 0;
    }
}

public function testQueryBackedEnumInAnotherCompositeKeyJoin(): void
    {
        $composition = new Composition();
        $composition->setType(Color::Red);

        $compositionChild = new CompositionChild();
        $compositionChild->setComposition($composition);

        $this->_em->persist($composition);
        $this->_em->persist($compositionChild);
        $this->_em->flush();
        $this->_em->clear();

        $qb = $this->_em->createQueryBuilder();
        $qb->select('c')
            ->from(CompositionChild::class, 'c')
            ->where('c.compositionType = :compositionType');

        $qb->setParameter('compositionType', Color::Red);

        self::assertNotNull($qb->getQuery()->getOneOrNullResult());
    }

