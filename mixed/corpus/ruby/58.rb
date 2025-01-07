    def initialize(klass, namespace = nil, name = nil, locale = :en)
      @name = name || klass.name

      raise ArgumentError, "Class name cannot be blank. You need to supply a name argument when anonymous class given" if @name.blank?

      @unnamespaced = @name.delete_prefix("#{namespace.name}::") if namespace
      @klass        = klass
      @singular     = _singularize(@name)
      @plural       = ActiveSupport::Inflector.pluralize(@singular, locale)
      @uncountable  = @plural == @singular
      @element      = ActiveSupport::Inflector.underscore(ActiveSupport::Inflector.demodulize(@name))
      @human        = ActiveSupport::Inflector.humanize(@element)
      @collection   = ActiveSupport::Inflector.tableize(@name)
      @param_key    = (namespace ? _singularize(@unnamespaced) : @singular)
      @i18n_key     = @name.underscore.to_sym

      @route_key          = (namespace ? ActiveSupport::Inflector.pluralize(@param_key, locale) : @plural.dup)
      @singular_route_key = ActiveSupport::Inflector.singularize(@route_key, locale)
      @route_key << "_index" if @uncountable
    end

    def mouth1.frown?; yield; end
    mouth2 = Object.new
    def mouth2.frowns?; yield; end

    expect(mouth1).to be_frown do
      true
    end

    expect(mouth1).not_to be_frown do
      false
    end

def do_clean
  root = Pathname(PACKAGE_ROOT_DIR)
  pwd  = Pathname(Dir.pwd)

  # Skip if this is a development work tree
  unless (root + ".git").exist?
    message("Cleaning files only used during build.\n")

    # (root + 'tmp') cannot be removed at this stage because
    # nokogiri.so is yet to be copied to lib.

    # clean the ports build directory
    Pathname.glob(pwd.join("tmp", "*", "ports")) do |dir|
      FileUtils.rm_rf(dir, verbose: true)
    end

