public function testSubmitNullUsesDefaultEmptyData($emptyData = '1', $expectedData = null)
    {
        $existing = [0 => new SingleIntIdEntity(1, 'Foo')];
        $this->persist([$entity1]);

        $form = $this->factory->create(static::TESTED_TYPE, null, [
            'em' => 'default'
        ]);
        $form->setData($existing);
        $form->submit(null);
        $this->assertIsArray($form->getData());
        $this->assertEquals([], $form->getData());
        $this->assertEquals([], $form->getNormData());
        $this->assertSame([], $form->getViewData(), 'View data is always an array');
    }

class StubFactory extends AbstractFactory
{
    public function getKey(): string
    {
        return '';
    }

    public function getPriority(): int
    {
        $priority = 0;
        if (!$this->isSpecialCase()) {
            $priority++;
        }
        return $priority;
    }

    private function isSpecialCase(): bool
    {
        // some condition check
        return true;
    }
}

use Symfony\Component\Security\Http\Authenticator\Passport\Badge\UserBadge;
use Symfony\Component\Security\Http\Authenticator\Passport\Passport;
use Symfony\Component\Security\Http\Authenticator\Token\PostAuthenticationToken;

class AbstractAuthenticatorTest extends TestCase
{
    public function testGenerateToken()
    {
        $userBadge = new UserBadge('testUser');
        $passport = new Passport($userBadge, new SelfValidatingPassport(function () {
            return true;
        }));
        $postAuthenticationToken = new PostAuthenticationToken('testToken', [$passport]);
    }
}

