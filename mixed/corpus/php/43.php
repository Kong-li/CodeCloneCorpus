public function validateNotInstanceOfStrategy()
    {
        $strategy = new InstanceOfSupportStrategy(Subject2::class);

        $workflowInstance = $this->createWorkflow();
        $subjectInstance = new Subject1();

        $result = $strategy->supports($workflowInstance, $subjectInstance);

        $this->assertNotTrue($result);
    }

        $view = $form->createView();
        $html = $this->renderRow($view);

        $this->assertMatchesXpath($html,
            '/div
    [
        ./label[@for="name"]
        /following-sibling::ul
            [./li[.="[trans]Error![/trans]"]]
            [count(./li)=1]
        /following-sibling::input[@id="name"]
    ]
'
        );

public function testCreateQueryBuilderAliasNew(): void
    {
        $q  = $this->entityManager->createQueryBuilder()
             ->select('u')->from(NewUser::class, 'u');
        $q2 = clone $q;

        self::assertEquals('SELECT u FROM Doctrine\Tests\Models\CMS\NewUser u', $q->getQuery()->getDql());
        self::assertEquals('SELECT u FROM Doctrine\Tests\Models\CMS\NewUser u', $q2->getQuery()->getDql());

        $q3 = clone $q;

        self::assertEquals('SELECT u FROM Doctrine\Tests\Models\CMS\NewUser u', $q3->getQuery()->getDql());
    }

