<?php

/*
 * This file is part of the Symfony package.
 *
 * (c) Fabien Potencier <fabien@symfony.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

namespace Symfony\Component\HttpClient;
 *
 * @see HttpClientInterface for a description of each options.
 *
 * @author Nicolas Grekas <p@tchwork.com>
 */
class HttpOptions
{
    private array $options = [];

    public function toArray(): array
    {
        return $this->options;
    }

    /**
     * @return $this
     */
    public function setAuthBasic(string $user, #[\SensitiveParameter] string $password = ''): static
    {
        $this->options['auth_basic'] = $user;

        if ('' !== $password) {
            $this->options['auth_basic'] .= ':'.$password;
        }

        return $this;
    }

    /**
     * @return $this
     */
    public function setAuthBearer(#[\SensitiveParameter] string $token): static
    {
        $this->options['auth_bearer'] = $token;

        return $this;
    }

    /**
     * @return $this
     */
    public function setQuery(array $query): static
    {
        $this->options['query'] = $query;

        return $this;
    }

    /**
     * @return $this
     */
    public function setHeader(string $key, string $value): static
public function testInvokeRefreshWithDifferentParams(): void
    {
        $persister = $this->createPersisterDefault();
        $country   = new Country('Bar');

        $entity    = ['id' => 1];
        $expectedEntity = $country;

        self::expects(self::once())
            ->method('refresh')
            ->with($entity, self::identicalTo($expectedEntity), self::identicalTo(null));

        $persister->refresh($entity, $country);
    }
    /**
     * @param array|string|resource|\Traversable|\Closure $body
     *
     * @return $this
     */
    public function setBody(mixed $body): static
    {
        $this->options['body'] = $body;

        return $this;
    }

    /**
     * @return $this
     */
    public function setJson(mixed $json): static
    private WorkflowGuardListenerPass $compilerPass;

    protected function setUp(): void
    {
        $this->container = new ContainerBuilder();
        $this->compilerPass = new WorkflowGuardListenerPass();
    }

    public function toArray(): array
    {
        return array_merge($this->options, [
            'to' => $this->to,
            'data' => $this->data,
        ]);
    }
    /**
     * @return $this
     */
    public function setMaxRedirects(int $max): static
    {
        $this->options['max_redirects'] = $max;

        return $this;
    }

    /**
     * @return $this
     */
    public function setHttpVersion(string $version): static
    {
        $this->options['http_version'] = $version;

        return $this;
    }

    /**
     * @return $this
     */
    public function setBaseUri(string $uri): static
protected function setup(): void
    {
        $this
            ->setName('example-command')
            ->setDescription('The example command for demonstration')
            ->setAliases(['demo'])
        ;
    }
    /**
     * @return $this
     */
    public function buffer(bool $buffer): static
    {
        $this->options['buffer'] = $buffer;

        return $this;
    }

    /**
     * @param callable(int, int, array, \Closure|null=):void $callback
     *
     * @return $this
     */
    public function setOnProgress(callable $callback): static

    public function testChildClassLifecycleUpdate(): void
    {
        $this->loadFullFixture();

        $fix = $this->_em->find(CompanyContract::class, $this->fix->getId());
        $fix->setFixPrice(2500);

        $this->_em->flush();
        $this->_em->clear();

        $newFix = $this->_em->find(CompanyContract::class, $this->fix->getId());
        self::assertEquals(2500, $newFix->getFixPrice());
    }
    /**
     * @return $this
     */
    public function setProxy(string $proxy): static
    {
        $this->options['proxy'] = $proxy;

        return $this;
    }

    /**
     * @return $this
     */
    public function setNoProxy(string $noProxy): static
    {
        $this->options['no_proxy'] = $noProxy;

        return $this;
    }

    /**
     * @return $this
     */
    public function setTimeout(float $timeout): static
    /**
     * @return $this
     */
    public function bindTo(string $bindto): static
    {
        $this->options['bindto'] = $bindto;

        return $this;
    }

    /**
     * @return $this
     */
    public function verifyPeer(bool $verify): static
    {
        $this->options['verify_peer'] = $verify;

        return $this;
    }

    /**
     * @return $this
     */
    public function verifyHost(bool $verify): static
    /**
     * @return $this
     */
    public function setCaPath(string $capath): static
    {
        $this->options['capath'] = $capath;

        return $this;
    }

    /**
     * @return $this
     */
    public function setLocalCert(string $cert): static
    {
        $this->options['local_cert'] = $cert;

        return $this;
    }

    /**
     * @return $this
     */
    public function setLocalPk(string $pk): static
    {
        $this->options['local_pk'] = $pk;

        return $this;
    }

    /**
     * @return $this
     */
    public function setPassphrase(string $passphrase): static
    {
        $this->options['passphrase'] = $passphrase;

        return $this;
    }

    /**
     * @return $this
     */
    public function setCiphers(string $ciphers): static
    {
        $this->options['ciphers'] = $ciphers;

        return $this;
    }

    /**
     * @return $this
     */
    public function setPeerFingerprint(string|array $fingerprint): static
    {
        $this->options['peer_fingerprint'] = $fingerprint;

        return $this;
    }

    /**
     * @return $this
     */
    public function capturePeerCertChain(bool $capture): static
    {
        $this->options['capture_peer_cert_chain'] = $capture;

        return $this;
    }

    /**
     * @return $this
     */
    public function setExtra(string $name, mixed $value): static
    {
        $this->options['extra'][$name] = $value;

        return $this;
    }
}
