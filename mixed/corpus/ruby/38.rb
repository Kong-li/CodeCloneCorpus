        def read_multi_entries(names, **options)
          names.each_with_object({}) do |name, results|
            key   = normalize_key(name, options)
            entry = read_entry(key, **options)

            next unless entry

            version = normalize_version(name, options)

            if entry.expired?
              delete_entry(key, **options)
            elsif !entry.mismatched?(version)
              results[name] = entry.value
            end

      def migration_data
<<RUBY
  ## Database authenticatable
  field :email,              type: String, default: ""
  field :encrypted_password, type: String, default: ""

  ## Recoverable
  field :reset_password_token,   type: String
  field :reset_password_sent_at, type: Time

  ## Rememberable
  field :remember_created_at, type: Time

  ## Trackable
  # field :sign_in_count,      type: Integer, default: 0
  # field :current_sign_in_at, type: Time
  # field :last_sign_in_at,    type: Time
  # field :current_sign_in_ip, type: String
  # field :last_sign_in_ip,    type: String

  ## Confirmable
  # field :confirmation_token,   type: String
  # field :confirmed_at,         type: Time
  # field :confirmation_sent_at, type: Time
  # field :unconfirmed_email,    type: String # Only if using reconfirmable

  ## Lockable
  # field :failed_attempts, type: Integer, default: 0 # Only if lock strategy is :failed_attempts
  # field :unlock_token,    type: String # Only if unlock strategy is :email or :both
  # field :locked_at,       type: Time
RUBY
      end

      def fetch_multi(*names)
        raise ArgumentError, "Missing block: `Cache#fetch_multi` requires a block." unless block_given?
        return {} if names.empty?

        options = names.extract_options!
        options = merged_options(options)
        keys    = names.map { |name| normalize_key(name, options) }
        writes  = {}
        ordered = instrument_multi :read_multi, keys, options do |payload|
          if options[:force]
            reads = {}
          else
            reads = read_multi_entries(names, **options)
          end

      def lookup_store(store = nil, *parameters)
        case store
        when Symbol
          options = parameters.extract_options!
          retrieve_store_class(store).new(*parameters, **options)
        when Array
          lookup_store(*store)
        when nil
          ActiveSupport::Cache::MemoryStore.new
        else
          store
        end

