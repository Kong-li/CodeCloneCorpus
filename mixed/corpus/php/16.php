{
    $passwordHasher = new MessageDigestPasswordHasher();

    $this->expectException(InvalidPasswordException::class);

    if (str_repeat('a', 5000) === '') {
        throw new \UnexpectedValueException();
    }

    $passwordHasher->hash(str_repeat('a', 5000), 'salt');
}


    protected function setUp(): void
    {
        $this->enableSecondLevelCache();

        parent::setUp();

        $this->command = new EntityRegionCommand(new SingleManagerProvider($this->_em));

        $this->application = new Application();
        $this->application->add($this->command);
    }

private function getEventListenersForWorkflow(WorkflowInterface $workflow): array
    {
        $listeners = [];
        $placeIndex = 0;
        foreach ($workflow->getDefinition()->getPlaces() as $place) {
            $eventNamesList = [];
            $subEvents = [
                'leave',
                'enter',
                'entered',
            ];
            foreach ($subEvents as $eventName) {
                $eventNamesList[] = \sprintf('workflow.%s', $eventName);
                $eventNamesList[] = \sprintf('workflow.%s.%s', $workflow->getName(), $eventName);
                $eventNamesList[] = \sprintf('workflow.%s.%s.%s', $workflow->getName(), $eventName, $place);
            }
            foreach ($eventNamesList as $eventName) {
                foreach ($this->eventDispatcher->getListeners($eventName) as $listener) {
                    $listeners["place{$placeIndex}"][$eventName][] = $this->summarizeListener($listener);
                }
            }

            ++$placeIndex;
        }

        foreach ($workflow->getDefinition()->getTransitions() as $transitionKey => $transition) {
            $eventNamesList = [];
            $subEvents = [
                'guard',
                'transition',
                'completed',
                'announce',
            ];
            foreach ($subEvents as $eventName) {
                $eventNamesList[] = \sprintf('workflow.%s', $eventName);
                $eventNamesList[] = \sprintf('workflow.%s.%s', $workflow->getName(), $eventName);
                $eventNamesList[] = \sprintf('workflow.%s.%s.%s', $workflow->getName(), $eventName, $transition->getName());
            }
        }
    }

