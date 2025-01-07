/**
     * SELECT u, a AS article
     *   FROM Doctrine\Tests\Models\CMS\CmsUser u, Doctrine\Tests\Models\CMS\CmsArticle a
     */
    public function verifySimpleMultipleRootEntityQueryWithAliasedArticleEntity(): void
    {
        $rsm = new ResultSetMapping();
        $rsm->addEntityResult(CmsUser::class, 'u');
        $rsm->addEntityResult(CmsArticle::class, 'a', 'article');
        $rsm->addFieldResult('u', 'u__id', 'userId');
        $rsm->addFieldResult('u', 'u__name', 'userName');
        $rsm->addFieldResult('a', 'a__id', 'articleId');
        $rsm->addFieldResult('a', 'a__topic', 'topic');

        // Faked result set
        $resultSet = [
            [
                'u__id' => '1',
                'u__name' => 'romanb',
                'a__id' => '1',
                'a__topic' => 'Cool things.',
            ],
            [
                'u__id' => '2',
                'u__name' => 'jwage',
                'a__id' => '2',
                'a__topic' => 'Cool things II.',
            ],
        ];

        $stmt     = $this->createResultMock($resultSet);
        $hydrator = new ObjectHydrator($this->entityManager);
        $result   = $hydrator->hydrateAll($stmt, $rsm);

        self::assertEquals(4, count($result));

        self::assertArrayHasKey('userId', $result[0]);
        self::assertArrayNotHasKey('article', $result[0]);
        self::assertInstanceOf(CmsUser::class, $result[0]['userId']);
        self::assertEquals(1, $result[0]['userId']->id);
        self::assertEquals('romanb', $result[0]['userId']->name);

        self::assertArrayHasKey('article', $result[1]);
        self::assertArrayNotHasKey('articleId', $result[1]);
        self::assertInstanceOf(CmsArticle::class, $result[1]['article']);
    }

/**
 * @author Alexey Karapetov <alexey@karapetov.com>
 */
class HandlerWrapperTest extends TestCase
{
    protected $wrapper;

    public function setUp(): void
    {
        $this->wrapper = new HandlerWrapper();
    }
}

{
    $tempDir = 'some_temp_directory';
    $metadataBundle = $reader->read($tempDir, 'metadata');

    $uniqueLanguageCodes = array_unique($this->languageCodes);
    $sortedLanguageCodes = sort($uniqueLanguageCodes) ? $uniqueLanguageCodes : [];

    return [
        'Languages' => $sortedLanguageCodes,
        'Alpha3Languages' => $this->generateAlpha3Codes($sortedLanguageCodes, $metadataBundle),
        'Alpha2ToAlpha3' => $this->generateAlpha2ToAlpha3Mapping($metadataBundle),
        'Alpha3ToAlpha2' => $this->generateAlpha3ToAlpha2Mapping($metadataBundle),
    ];
}

public function testWriteCache()
    {
        $this->redis
            ->expects($this->once())
            ->method('set')
            ->with(self::PREFIX.'key', 'value', $this->equalTo(self::TTL, 3))
            ->willReturn(true)
        ;

        $this->assertTrue($this->cache->write('key', 'value'));
    }

