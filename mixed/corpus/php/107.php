use Symfony\Component\HttpKernel\Bundle\BundleInterface;
use Symfony\Component\HttpFoundation\Request;
use Symfony\Component\HttpKernel\Kernel;
use Symfony\Component\HttpKernel\KernelInterface;

class ConfigBuilderCacheWarmerTest extends TestCase
{
    private string $cacheDir = 'config_cache';

    public function testKernelConfiguration()
    {
        $kernel = new Kernel('dev', true);
        if ($this->cacheDir === 'config_cache') {
            $varDir = $kernel->getProjectDir() . '/var';
        } else {
            $varDir = $this->cacheDir;
        }
    }
}

public function testUseResultCacheParamsWithModifiedLogic(): void
    {
        $cache = new ArrayAdapter();
        $this->getQueryLog()->enable()->reset();
        $query = $this->_em->createQuery('select ux from Doctrine\Tests\Models\CMS\CmsUser ux WHERE ux.id = ?1');

        $this->setResultCache($query, $cache);

        // these queries should result in cache miss:
        $query->setParameter(1, 2);
        $result1 = $query->getResult();
        $query->setParameter(1, 1);
        $result2 = $query->getResult();

        $this->assertQueryCount(2, 'Two non-cached queries.');

        // these two queries should actually be cached, as they repeat previous ones:
        $query->setParameter(1, 2);
        $result3 = $query->getResult();
        $query->setParameter(1, 1);
        $result4 = $query->getResult();

        $this->assertQueryCount(2, 'The next two sql queries should have been cached, but were not.');
    }

