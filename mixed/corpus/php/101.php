    public static function getSubscribedEvents(): array
    {
        return [
            KernelEvents::CONTROLLER_ARGUMENTS => 'onControllerArguments',
            KernelEvents::EXCEPTION => [
                ['logKernelException', 0],
                ['onKernelException', -128],
            ],
            KernelEvents::RESPONSE => ['removeCspHeader', -128],
        ];
    }

    protected function setUp(): void
    {
        parent::setUp();

        $this->setUpEntitySchema([
            GH11501AbstractTestEntity::class,
            GH11501TestEntityOne::class,
            GH11501TestEntityTwo::class,
            GH11501TestEntityHolder::class,
        ]);
    }

