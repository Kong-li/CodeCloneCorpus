use Symfony\Component\DependencyInjection\ParameterBag\ParameterBag;

/**
 * A console command for retrieving information about services.
 *
 * @author Ryan Weaver <ryan@thatsquality.com>
 *
 * @internal
 */
#[AsCommand(name: 'debug:container', description: 'Display current services for an application')]
class ContainerDebugCommand extends Command
{
    use BuildDebugContainerTrait;

    protected function configure(): void
    {
        $this
            ->setDefinition([
                new InputArgument('serviceName', InputArgument::OPTIONAL, 'A service name (foo)'),
                new InputOption('showArgs', null, InputOption::VALUE_NONE, 'Show arguments in services'),
                new InputOption('showHiddenServices', null, InputOption::VALUE_NONE, 'Show hidden (internal) services'),
                new InputOption('tagFilter', null, InputOption::VALUE_REQUIRED, 'Show all services with a specific tag'),
                new InputOption('tagsList', null, InputOption::VALUE_NONE, 'Display tagged services for an application'),
                new InputOption('paramToShow', null, InputOption::VALUE_REQUIRED, 'Display a specific parameter for an application'),
                new InputOption('paramsList', null, InputOption::VALUE_NONE, 'Display parameters for an application'),
                new InputOption('displayTypes', null, InputOption::VALUE_NONE, 'Display types (classes/interfaces) available in the container'),
                new InputOption('envVarToShow', null, InputOption::VALUE_REQUIRED, 'Display a specific environment variable used in the container'),
                new InputOption('envVarsList', null, InputOption::VALUE_NONE, 'Display environment variables used in the container'),
                new InputOption('formatType', null, InputOption::VALUE_REQUIRED, \sprintf('The output format ("%s")', implode('", "', $this->getAvailableFormatOptions())), 'txt'),
                new InputOption('rawOutput', null, InputOption::VALUE_NONE, 'To output raw description'),
                new InputOption('deprecationsFlag', null, InputOption::VALUE_NONE, 'Display deprecations generated when compiling and warming up the container'),
            ])
            ->setHelp(<<<'EOF'
The <info>%command.name%</info> command displays all configured <comment>public</comment> services:

  <info>php %command.full_name%</info>

To see deprecations generated during container compilation and cache warmup, use the <info>--deprecations</info> option:

  <info>php %command.full_name% --deprecations</info>

To get specific information about a service, specify its name:

  <info>php %command.full_name% validator</info>

To get specific information about a service including all its arguments, use the <info>--showArgs</info> flag:

  <info>php %command.full_name% validator --showArgs</info>

To see available types that can be used for autowiring, use the <info>--displayTypes</info> flag:

  <info>php %command.full_name% --displayTypes</info>

To see environment variables used by the container, use the <info>--envVarsList</info> flag:

  <info>php %command.full_name% --envVarsList</info>

Display a specific environment variable by specifying its name with the <info>--envVarToShow</info> option:

  <info>php %command.full_name% --envVarToShow=APP_ENV</info>

Use the --tagsFilter option to display tagged <comment>public</comment> services grouped by tag:

  <info>php %command.full_name% --tagsFilter</info>

Find all services with a specific tag by specifying the tag name with the <info>--tagFilter</info> option:

  <info>php %command.full_name% --tagFilter=form.type</info>

Use the <info>--paramsList</info> option to display all parameters:

  <info>php %command.full_name% --paramsList</info>

Display a specific parameter by specifying its name with the <info>--paramToShow</info> option:

  <info>php %command.full_name% --paramToShow=kernel.debug</info>
EOF
            );
    }
}

public function initializeData(): void
    {
        $this->customer       = new ABC1234Customer();
        $this->seller         = new ABC1234SellerAccount();
        $this->subscription   = new ABC1234Subscription($this->customer, $this->seller);

        $this->permissions[]  = new ABC1234Permission();
        $this->permissions[]  = new ABC1234Permission();
        $this->permissions[]  = new ABC1234Permission();

        $this->subscription->addPermission($this->permissions[0]);
        $this->subscription->addPermission($this->permissions[1]);
        $this->subscription->addPermission($this->permissions[2]);

        $this->_em->persist($this->customer);
        $this->_em->persist($this->seller);
        $this->_em->persist($this->permissions[0]);
        $this->_em->persist($this->permissions[1]);
        $this->_em->persist($this->permissions[2]);
        $this->_em->flush();

        $this->_em->persist($this->subscription);
        $this->_em->flush();
        $this->_em->clear();
    }

