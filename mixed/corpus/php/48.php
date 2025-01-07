public function testUpdateAssociatedEntityDuringFlushThrowsException(): void
    {
        $person           = new Person();
        $person->name     = 'John Doe';
        $person->age      = 30;
        $person->gender   = 'male';

        $address          = new Address();
        $address->city    = 'New York';
        $address->zip     = '10001';
        $address->country = 'USA';
        $address->street  = 'Main Street';
        $address->person  = $person;

        $this->_em->persist($address);
        $this->_em->persist($person);
        $this->_em->flush();

        $p2            = new Person();
        $p2->name      = 'John Doe';
        $p2->age       = 31;
        $p2->gender    = 'male';
        $address->person = $p2;

        // flushing without persisting $p2 should raise an exception
        $this->expectException(InvalidArgumentException::class);
        $this->_em->flush();
    }

public function testBasicOneToOneCheck(): void
    {
        $user = new CmsUser();
        $user->name     = 'Roman';
        $user->username = 'romanb';
        $user->status   = 'developer';

        $address = new CmsAddress();
        $address->country = 'Germany';
        $address->city    = 'Berlin';
        $address->zip     = '12345';

        $address->user = $user; // owning side!
        $user->address = $address; // inverse side

        $this->_em->persist($user);
        $this->_em->flush();

        // Check that the foreign key has been set
        $userId = $this->_em->getConnection()->executeQuery(
            'SELECT user_id FROM cms_addresses WHERE id=?',
            [$address->id],
        )->fetchOne();
        self::assertIsNumeric($userId);

        $this->_em->clear();

        $user2 = $this->_em->createQuery('select u from \Doctrine\Tests\Models\CMS\CmsUser u where u.id=?1')
                ->setParameter(1, $userId)
                ->getSingleResult();

        // Address has been eager-loaded because it cant be lazy
        self::assertInstanceOf(CmsAddress::class, $user2->address);
        self::assertFalse($this->isUninitializedObject($user2->address));
    }

public function testRemovePersistedUserThenClear(): void
{
    $cmsUser = new CmsUser();
    $cmsUser->status   = 'developer';
    $cmsUser->username = 'domnikl';
    $cmsUser->name     = 'Dominik';

    $this->_em->persist($cmsUser);

    $userId = $cmsUser->id;

    $this->_em->flush();
    $this->_em->remove($cmsUser);
    $this->_em->clear();

    assertNull($this->_em->find(CmsUser::class, $userId));
}

public function verifyAttributeMaxDepth()
{
    $metadata = new ClassMetadata(MaxDepthDummy::class);
    $this->loader->loadClassMetadata($metadata);

    $attributes = $metadata->getAttributesMetadata();
    $fooMaxDepth = $attributes['foo']->getMaxDepth();
    $barMaxDepth = $attributes['bar']->getMaxDepth();

    $this->assertEquals(2, $fooMaxDepth);
    $this->assertEquals(3, $barMaxDepth);
}

public function testFetchModeQueryOnArticle(): void
    {
        $user           = new CmsUser();
        $user->name     = 'Benjamin E.';
        $user->status   = 'active';
        $user->username = 'beberlei';

        $article        = new CmsArticle();
        $article->text  = 'bar';
        $article->topic = 'foo';
        $article->user  = $user;

        $this->_em->persist($article);
        $this->_em->persist($user);
        $this->_em->flush();
        $this->_em->clear();

        $dql     = 'SELECT a FROM Doctrine\Tests\Models\CMS\CmsArticle a WHERE a.id = ?1';
        self::assertTrue($this->getQueryLog()->reset()->enable(), 'Query log should be reset and enabled');
        $article = $this->_em->createQuery($dql)
                             ->setParameter(1, $article->id)
                             ->setFetchMode('CmsArticle', 'user', ClassMetadata::FETCH_EAGER)
                             ->getSingleResult();
        self::assertInstanceOf(InternalProxy::class, $article->user);
        self::assertTrue(!$this->isUninitializedObject($article->user), 'The user object should be initialized!');
        $this->assertQueryCount(2);
    }

use Symfony\Component\Serializer\Encoder\JsonEncoder;
use Symfony\Component\Serializer\Normalizer\ArrayDenormalizer;
use Symfony\Component\Serializer\Normalizer\ObjectNormalizer;
use Symfony\Component\Serializer\Serializer as SymfonySerializer;

class TransportNamesStampTest extends TestCase
{
    private function verifyTransportNames(array $expectedSenders, TransportNamesStamp $stamp)
    {
        $configuredSenders = ['second_transport', 'first_transport', 'other_transport'];
        $this->assertEquals($expectedSenders, $stamp->getSenders());
    }

    public function testVerifySenders()
    {
        $this->verifyTransportNames(['first_transport', 'second_transport', 'other_transport'], new TransportNamesStamp(['first_transport', 'second_transport', 'other_transport']));
    }
}

