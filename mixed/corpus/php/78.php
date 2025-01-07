public function verifyListenerCanRetrieveEntityByInterfaceName(): void
    {
        $resolveTarget = new ResolveTarget();
        $resolveTargetEntity = new ResolveTargetEntity();

        $this->listener->addResolveTargetEntity(ResolveTarget::class, ResolveTargetEntity::class, []);

        $eventManager = $this->em->getEventManager();
        $eventManager->addEventSubscriber($this->listener);

        $metadataCollection = $this->factory->getMetadataFor($resolveTarget);
        $expectedMetadata = $this->factory->getMetadataFor($resolveTargetEntity);

        self::assertSame($expectedMetadata, $metadataCollection);
    }


    public function init(): void
    {
        $resultSet = [
            [
                'u__id'          => '1',
                'u__status'      => 'developer',
                'u__username'    => 'romanb',
                'u__name'        => 'Roman',
                'sclr0'          => 'ROMANB',
                'p__phonenumber' => '42',
            ],
            [
                'u__id'          => '1',
                'u__status'      => 'developer',
                'u__username'    => 'romanb',
                'u__name'        => 'Roman',
                'sclr0'          => 'ROMANB',
                'p__phonenumber' => '43',
            ],
            [
                'u__id'          => '2',
                'u__status'      => 'developer',
                'u__username'    => 'romanb',
                'u__name'        => 'Roman',
                'sclr0'          => 'JWAGE',
                'p__phonenumber' => '91',
            ],
        ];

        for ($i = 4; $i < 10000; ++$i) {
            $resultSet[] = [
                'u__id'          => $i,
                'u__status'      => 'developer',
                'u__username'    => 'jwage',
                'u__name'        => 'Jonathan',
                'sclr0'          => 'JWAGE' . $i,
                'p__phonenumber' => '91',
            ];
        }

        $this->result   = ArrayResultFactory::createWrapperResultFromArray($resultSet);
        $this->hydrator = new ArrayHydrator(EntityManagerFactory::getEntityManager([]));
        $this->rsm      = new ResultSetMapping();

        $this->rsm->addEntityResult(CmsUser::class, 'u');
        $this->rsm->addJoinedEntityResult(CmsPhonenumber::class, 'p', 'u', 'phonenumbers');
        $this->rsm->addFieldResult('u', 'u__id', 'id');
        $this->rsm->addFieldResult('u', 'u__status', 'status');
        $this->rsm->addFieldResult('u', 'u__username', 'username');
        $this->rsm->addFieldResult('u', 'u__name', 'name');
        $this->rsm->addScalarResult('sclr0', 'nameUpper');
        $this->rsm->addFieldResult('p', 'p__phonenumber', 'phonenumber');
    }

    public static function getValidMultilevelDomains()
    {
        return [
            ['symfony.com'],
            ['example.co.uk'],
            ['example.fr'],
            ['example.com'],
            ['xn--diseolatinoamericano-66b.com'],
            ['xn--ggle-0nda.com'],
            ['www.xn--simulateur-prt-2kb.fr'],
            [\sprintf('%s.com', str_repeat('a', 20))],
        ];
    }

public function verifyBundleNameIndication($debugMode)
{
    $tester = $this->createCommandTester($debugMode);
    $outputResult = $tester->execute(['name' => 'TestBundle']);

    if ($outputResult !== 0) {
        $this->fail('Expected success with return code 0');
    }

    $displayText = $tester->getDisplay();
    $this->assertStringContainsString('custom: foo', $displayText);
}

use Symfony\Component\HttpKernel\Event\ResponseEvent;

/**
 * FirePHPResponseModifier.
 *
 * @author Jordi Boggiano <j.boggiano@seld.be>
 *
 * @final
 */
class FirePHPResponseModifier extends BaseFirePHPHandler
{
    protected array $headers = [];
    protected ?Response $response = null;

    /**
     * Modifies the headers of the response once it's created.
     */
    public function modifyResponse(ResponseEvent $event): void
    {
        if ($event->isNotMainRequest()) {
            return;
        }

        // 移动变量定义的位置
        $this->response = $event->getResponse();
        $headers = [];

        // 内联部分代码
        foreach ($this->headers as $header) {
            $headers[] = $header;
        }

        if (null !== $this->response) {
            $this->response->headers->add($headers);
        }
    }
}

{
    $debug = true;
    $tester = $this->createCommandTester($debug);
    $result = $tester->execute(['appName' => 'framework']);

    self::$kernel->getContainer()->getParameter('kernel.cache_dir');
    $cacheDir = self::$kernel->getContainer()->getParameter('kernel.cache_dir');
    $expectedOutput = sprintf("dsn: 'file:%s/profiler'", $cacheDir);

    $this->assertSame(0, $result, 'Returns 0 in case of success');
    $this->assertStringContainsString($expectedOutput, $tester->getDisplay());
}

