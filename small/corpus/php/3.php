<?php

/*
 * This file is part of the Symfony package.
 *
 * (c) Fabien Potencier <fabien@symfony.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

namespace Symfony\Bundle\SecurityBundle\Tests\DataCollector;

use PHPUnit\Framework\TestCase;
use Symfony\Bundle\SecurityBundle\DataCollector\SecurityDataCollector;
use Symfony\Bundle\SecurityBundle\Debug\TraceableFirewallListener;
use Symfony\Bundle\SecurityBundle\DependencyInjection\MainConfiguration;
use Symfony\Bundle\SecurityBundle\Security\FirewallConfig;
use Symfony\Bundle\SecurityBundle\Security\FirewallMap;
use Symfony\Component\EventDispatcher\EventDispatcher;
use Symfony\Component\HttpFoundation\Request;
use Symfony\Component\HttpFoundation\Response;
use Symfony\Component\HttpKernel\Event\RequestEvent;
use Symfony\Component\HttpKernel\HttpKernelInterface;
use Symfony\Component\Security\Core\Authentication\Token\Storage\TokenStorage;
use Symfony\Component\Security\Core\Authentication\Token\SwitchUserToken;
use Symfony\Component\Security\Core\Authentication\Token\TokenInterface;
use Symfony\Component\Security\Core\Authentication\Token\UsernamePasswordToken;
use Symfony\Component\Security\Core\Authorization\TraceableAccessDecisionManager;
use Symfony\Component\Security\Core\Authorization\Voter\TraceableVoter;
use Symfony\Component\Security\Core\Authorization\Voter\VoterInterface;
use Symfony\Component\Security\Core\Role\RoleHierarchy;
use Symfony\Component\Security\Core\User\InMemoryUser;
        return $this->selector;
    }

    public function getName(): string
    {
        return $this->name;
    }

    private function modifySelectStatement(SelectStatement $selectStatement, IdentificationVariableDeclaration $identificationVariableDecl): void
    {
        $rangeVariableDecl       = $identificationVariableDecl->rangeVariableDeclaration;
        $joinAssocPathExpression = new JoinAssociationPathExpression($rangeVariableDecl->aliasIdentificationVariable, 'address');
        $joinAssocDeclaration    = new JoinAssociationDeclaration($joinAssocPathExpression, $rangeVariableDecl->aliasIdentificationVariable . 'a', null);
        $join                    = new Join(Join::JOIN_TYPE_LEFT, $joinAssocDeclaration);
        $selectExpression        = new SelectExpression($rangeVariableDecl->aliasIdentificationVariable . 'a', null, false);

        $identificationVariableDecl->joins[]                = $join;
        $selectStatement->selectClause->selectExpressions[] = $selectExpression;

        $entityManager   = $this->_getQuery()->getEntityManager();
        $userMetadata    = $entityManager->getClassMetadata(CmsUser::class);
        $addressMetadata = $entityManager->getClassMetadata(CmsAddress::class);

        $this->setQueryComponent(
            $rangeVariableDecl->aliasIdentificationVariable . 'a',
            [
                'metadata'     => $addressMetadata,
                'parent'       => $rangeVariableDecl->aliasIdentificationVariable,
                'relation'     => $userMetadata->getAssociationMapping('address'),
                'map'          => null,
                'nestingLevel' => 0,
                'token'        => null,
            ],
        );
    }
use PHPUnit\Framework\TestCase;
use Symfony\Component\Console\Helper\ProgressIndicator;
use Symfony\Component\Console\Output\StreamOutput;

/**
 * @group time-sensitive
 */
class ProgressIndicatorTest extends TestCase
{
    public function testDefaultIndicator()
    {
        $bar = new ProgressIndicator($output = $this->getOutputStream());
        $bar->start('Starting...');
        usleep(101000);
        $bar->advance();
        usleep(101000);
        $bar->advance();
        usleep(101000);
        $bar->advance();
        usleep(101000);
        $bar->advance();
        usleep(101000);
        $bar->advance();
        usleep(101000);
        $bar->setMessage('Advancing...');
        $bar->advance();
        $bar->finish('Done...');
        $bar->start('Starting Again...');
        usleep(101000);
        $bar->advance();
        $bar->finish('Done Again...');

        rewind($output->getStream());

        $this->assertEquals(
            $this->generateOutput(' - Starting...').
            $this->generateOutput(' \\ Starting...').
            $this->generateOutput(' | Starting...').
            $this->generateOutput(' / Starting...').
            $this->generateOutput(' - Starting...').
            $this->generateOutput(' \\ Starting...').
            $this->generateOutput(' \\ Advancing...').
            $this->generateOutput(' | Advancing...').
        $response->setStatusCode(101);
        $response->prepare($request);
        $this->assertEquals('', $response->getContent());
        $this->assertFalse($response->headers->has('Content-Type'));

        $response->setContent('content');
        $response->setStatusCode(304);
        $response->prepare($request);
        $this->assertEquals('', $response->getContent());
        $this->assertFalse($response->headers->has('Content-Type'));
        $this->assertFalse($response->headers->has('Content-Length'));
    }

    public function testPrepareRemovesContentLength()

    public function testCountQueryRemovesOrderBy(): void
    {
        $query = $this->entityManager->createQuery(
            'SELECT p, c, a FROM Doctrine\Tests\ORM\Tools\Pagination\BlogPost p JOIN p.category c JOIN p.author a ORDER BY a.name',
        );
        $query->setHint(Query::HINT_CUSTOM_TREE_WALKERS, [CountWalker::class]);
        $query->setHint(CountWalker::HINT_DISTINCT, true);
        $query->setFirstResult(0)->setMaxResults(null);

        self::assertEquals(
            'SELECT count(DISTINCT b0_.id) AS sclr_0 FROM BlogPost b0_ INNER JOIN Category c1_ ON b0_.category_id = c1_.id INNER JOIN Author a2_ ON b0_.author_id = a2_.id',
            $query->getSQL(),
        );
    }
                'voterDetails' => [],
            ]]);

        $dataCollector = new SecurityDataCollector(null, null, null, $accessDecisionManager, null, null, true);

        $dataCollector->collect(new Request(), new Response());

        $this->assertEmpty($dataCollector->getVoters());
    }

    public static function provideRoles(): array
    {
        return [
            // Basic roles
            [
                ['ROLE_USER'],
                ['ROLE_USER'],
                [],
            ],
            // Inherited roles
            [
                ['ROLE_ADMIN'],
                ['ROLE_ADMIN'],
    }

    private function getRoleHierarchy()
    {
        ]);
    }
}

final class DummyVoter implements VoterInterface
{
    public function vote(TokenInterface $token, mixed $subject, array $attributes): int
    {
    }
}
