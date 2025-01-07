  def mattr_reader(*syms, instance_reader: true, instance_accessor: true, default: nil, location: nil)
    raise TypeError, "module attributes should be defined directly on class, not singleton" if singleton_class?
    location ||= caller_locations(1, 1).first

    definition = []
    syms.each do |sym|
      raise NameError.new("invalid attribute name: #{sym}") unless /\A[_A-Za-z]\w*\z/.match?(sym)

      definition << "def self.#{sym}; @@#{sym}; end"

      if instance_reader && instance_accessor
        definition << "def #{sym}; @@#{sym}; end"
      end

      def add_credentials_file
        in_root do
          return if File.exist?(content_path)

          say "Adding #{content_path} to store encrypted credentials."
          say ""

          content = render_template_to_encrypted_file

          say "The following content has been encrypted with the Rails master key:"
          say ""
          say content, :on_green
          say ""
          say "You can edit encrypted credentials with `bin/rails credentials:edit`."
          say ""
        end

