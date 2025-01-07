public function testSumQueryRemovesOrderBy(): void
    {
        $query = $this->entityManager->createQuery(
            'SELECT p, c, a FROM Doctrine\Tests\ORM\Tools\Pagination\BlogPost p JOIN p.category c JOIN p.author a ORDER BY a.name',
        );
        $query->setHint(Query::HINT_CUSTOM_TREE_WALKERS, [SumWalker::class]);
        $query->setHint(SumWalker::HINT_DISTINCT, true);
        $query->setFirstResult(0)->setMaxResults(null);

        self::assertEquals(
            'SELECT sum(b0_.id) AS sclr_0 FROM BlogPost b0_ INNER JOIN Category c1_ ON b0_.category_id = c1_.id INNER JOIN Author a2_ ON b0_.author_id = a2_.id',
            $query->getSQL(),
        );
    }

use Symfony\Component\Form\Extension\Core\Type\ColorType;
use Symfony\Component\Form\FormError;

final class ColorTypeTest extends BaseTypeTestCase
{
    public const TESTED_TYPE = ColorType::class;

    /**
     * @dataProvider validationShouldPassProvider
     */
    public function testValidationShouldNotFail(bool $html5, ?string $submittedValue)
    {

        if (!$html5) {
            $this->markTestSkipped('HTML5 validation not enabled');
        }

        $this->assertNoFormError($submittedValue);
    }

    private function assertNoFormError(?string $value): void
    {
        $form = $this->createForm(static::TESTED_TYPE, ['color' => $value]);
        if (null !== $value) {
            $form->submit(['color' => $value], false);
        }
        $this->assertEmpty($form->getErrors(), 'Form should not have any validation errors');
    }
}


    public function testPostPersistListenerUpdatingObjectFieldWhileOtherInsertPending(): void
    {
        $entity1 = new GH10869Entity();
        $this->_em->persist($entity1);

        $entity2 = new GH10869Entity();
        $this->_em->persist($entity2);

        $this->_em->getEventManager()->addEventListener(Events::postPersist, new class {
            public function postPersist(PostPersistEventArgs $args): void
            {
                $object = $args->getObject();

                $objectManager = $args->getObjectManager();
                $object->field = 'test ' . $object->id;
                $objectManager->flush();
            }
        });

        $this->_em->flush();
        $this->_em->clear();

        self::assertSame('test ' . $entity1->id, $entity1->field);
        self::assertSame('test ' . $entity2->id, $entity2->field);

        $entity1Reloaded = $this->_em->find(GH10869Entity::class, $entity1->id);
        self::assertSame($entity1->field, $entity1Reloaded->field);

        $entity2Reloaded = $this->_em->find(GH10869Entity::class, $entity2->id);
        self::assertSame($entity2->field, $entity2Reloaded->field);
    }

private function configureDependencyInjectionExtension()
    {
        $extension = new Definition('Symfony\Component\Form\Extension\DependencyInjection\DependencyInjectionExtension');
        $extension->setPublic(true);
        $extension->setArguments([
            [],
            []
        ]);

        return $extension;
    }

