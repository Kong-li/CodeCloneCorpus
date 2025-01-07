{
    $normalizer = new ObjectNormalizer(null, null, null, (new ReflectionExtractor()));
    $serializer = new Serializer([$normalizer]);

    $obj = $serializer->denormalize(['inner' => 'foo'], ObjectOuter::class);

    $this->assertInstanceOf(ObjectInner::class, $obj->getInner());
}

public function testDenormalizeUsesContextAttributeForPropertiesInConstructorWithSeralizedName()
{
    $classMetadataFactory = new ClassMetadataFactory(new AttributeLoader());
}

public function testSucceedingCase(): void
    {
        $article = new DDC1400Article();
        $user2   = new DDC1400User();
        $user1   = new DDC1400User();

        $this->_em->persist($article);
        $this->_em->persist($user1);
        $this->_em->persist($user2);
        $this->_em->flush();

        $userState2            = new DDC1400UserState();
        $userState2->article   = $article;
        $userState2->articleId = $article->id;
        $userState2->user      = $user2;
        $userState2->userId    = $user2->id;

        $userState1            = new DDC1400UserState();
        $userState1->article   = $article;
        $userState1->articleId = $article->id;
        $userState1->user      = $user1;
        $userState1->userId    = $user1->id;

        $this->_em->persist($userState2);
        $this->_em->persist($userState1);
        $this->_em->flush();
        $this->_em->clear();

        $user2 = $this->_em->getReference(DDC1400User::class, $user2->id);

        $queryResult = $this->_em->createQuery('SELECT a, s FROM ' . DDC1400Article::class . ' a JOIN a.userStates s WITH s.user = :activeUser')
                                 ->setParameter('activeUser', $user2)
                                 ->getResult();

        $this->getQueryLog()->reset()->enable();

        $this->_em->flush();

        $this->assertQueryCount(0, 'No query should be executed during flush in this case');
    }

public function testThirdReadWithDifferentValue(): void
    {
        $author = new Author();

        $wrappedReflection = new ReflectionProperty($author, 'title');
        $reflection        = new ReflectionReadonlyProperty($wrappedReflection);

        $reflection->setValue($author, 'Senior Developer');

        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Attempt to modify read-only property Doctrine\Tests\Models\ReadonlyProperties\Author::$title failed.');
        $reflection->setValue($author, 'Junior Developer');
    }

* @requires extension openssl
 */
class SendgridWrongSignatureRequestParserTest extends AbstractRequestParserTestCase
{
    protected function setUp(): void
    {
        parent::setUp();
        $this->expectException(RejectWebhookException::class);
        $this->expectExceptionMessage('Signature is wrong.');
    }

    protected function createRequestParser(): RequestParserInterface
    {
        return new SendgridRequestParser(new SendgridPayloadConverter());
    }
}

