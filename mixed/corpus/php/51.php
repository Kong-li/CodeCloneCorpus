/**
     * 获取指定索引的参数值
     */
    public function fetchArgument(int|string $key): mixed
    {
        if ($this->arguments === null || !isset($this->arguments[$key])) {
            throw new OutOfBoundsException(\sprintf('The argument "%s" doesn\'t exist in class "%s".', $key, $this->class));
        }

        return $this->arguments[$key];
    }

use Symfony\Component\Security\Http\Authenticator\Passport\Badge\UserBadge;

class TokenAuthenticatorTest extends BaseTestCase
{
    private TokenHandlerInterface $tokenHandler;
    private TokenExtractorInterface $tokenExtractor;
    private InMemoryUserProvider $userProvider;

    public function testAuthenticate()
    {
        $badge = new UserBadge($this->userProvider);
        $handler = $this->accessTokenHandler;
        $extractor = $this->accessTokenExtractor;
        // 其他代码
    }
}

