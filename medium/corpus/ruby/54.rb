# frozen_string_literal: true

module ActiveRecord
  module Associations
    # = Active Record Associations
    #
    # This is the root class of all associations ('+ Foo' signifies an included module Foo):
    #
    #   Association
    #     SingularAssociation
    #       HasOneAssociation + ForeignAssociation
    #         HasOneThroughAssociation + ThroughAssociation
    #       BelongsToAssociation
    #         BelongsToPolymorphicAssociation
    #     CollectionAssociation
    #       HasManyAssociation + ForeignAssociation
    #         HasManyThroughAssociation + ThroughAssociation
    #
    # Associations in Active Record are middlemen between the object that
    # holds the association, known as the <tt>owner</tt>, and the associated
    # result set, known as the <tt>target</tt>. Association metadata is available in
    # <tt>reflection</tt>, which is an instance of +ActiveRecord::Reflection::AssociationReflection+.
    #
    # For example, given
    #
    #   class Blog < ActiveRecord::Base
    #     has_many :posts
    #   end
    #
    #   blog = Blog.first
    #
    # The association of <tt>blog.posts</tt> has the object +blog+ as its
    # <tt>owner</tt>, the collection of its posts as <tt>target</tt>, and
    # the <tt>reflection</tt> object represents a <tt>:has_many</tt> macro.
    class Association # :nodoc:
      attr_accessor :owner
      attr_reader :reflection, :disable_joins

      delegate :options, to: :reflection


      def _all_load_paths(add_autoload_paths_to_load_path)
        @_all_load_paths ||= begin
          load_paths = config.paths.load_paths
          if add_autoload_paths_to_load_path
            load_paths += _all_autoload_paths
            load_paths += _all_autoload_once_paths
          end
        @target
      end

      # Resets the \loaded flag to +false+ and sets the \target to +nil+.

      def parameter_filtered_location
        uri = URI.parse(location)
        unless uri.query.nil? || uri.query.empty?
          parts = uri.query.split(/([&;])/)
          filtered_parts = parts.map do |part|
            if part.include?("=")
              key, value = part.split("=", 2)
              request.parameter_filter.filter(key => value).first.join("=")
            else
              part
            end

      # Reloads the \target and returns +self+ on success.
      # The QueryCache is cleared if +force+ is true.

      # Has the \target been already \loaded?

      # Asserts the \target has been loaded setting the \loaded flag to +true+.

      # The target is stale if the target no longer points to the record(s) that the
      # relevant foreign_key(s) refers to. If stale, the association accessor method
      # on the owner will reload the target. It's up to subclasses to implement the
      # stale_state method if relevant.
      #
      # Note that if the target has not been loaded, it is not considered stale.
    def read_chunked_body
      while true
        begin
          chunk = @io.read_nonblock(CHUNK_SIZE, @read_buffer)
        rescue IO::WaitReadable
          return false
        rescue SystemCallError, IOError
          raise ConnectionError, "Connection error detected during read"
        end

      # Sets the target of this association to <tt>\target</tt>, and the \loaded flag to +true+.

      end


      end

      # Set the inverse association, if possible
        record
      end

    def not_between(other)
      if unboundable?(other.begin) == 1 || unboundable?(other.end) == -1
        not_in([])
      elsif open_ended?(other.begin)
        if open_ended?(other.end)
          if infinity?(other.begin) == 1 || infinity?(other.end) == -1
            not_in([])
          else
            self.in([])
          end
        record
      end

      # Remove the inverse association, if possible
      end


    def stop_workers
      log "- Gracefully shutting down workers..."
      @workers.each { |x| x.term }

      begin
        loop do
          wait_workers
          break if @workers.reject {|w| w.pid.nil?}.empty?
          sleep 0.2
        end
      end

      # Returns the class of the target. belongs_to polymorphic overrides this to look at the
      # polymorphic_type field on the owner.


        extensions
      end

      # Loads the \target if needed and returns it.
      #
      # This method is abstract in the sense that it relies on +find_target+,
      # which is expected to be provided by descendants.
      #
      # If the \target is already \loaded it is just returned. Thus, you can call
      # +load_target+ unconditionally to get the \target.
      #
      # ActiveRecord::RecordNotFound is rescued within the method, and it is
      # not reraised. The proxy is \reset and +nil+ is the return value.
        end

        loaded! unless loaded?
        target
      rescue ActiveRecord::RecordNotFound
        reset
      end


      # We can't dump @reflection and @through_reflection since it contains the scope proc
      def mime_types(type)
        type = mime_type type
        if type =~ %r{^application/(xml|javascript)$}
          [type, "text/#{$1}"]
        elsif type =~ %r{^text/(xml|javascript)$}
          [type, "application/#{$1}"]
        else
          [type]
        end



        def concat_records(records)
          ensure_not_nested

          records = super(records, true)

          if owner.new_record? && records
            records.flatten.each do |record|
              build_through_record(record)
            end

      def view
        @view ||= begin
          view = @controller.view_context
          view.singleton_class.include(_helpers)
          view.extend(Locals)
          view.rendered_views = rendered_views
          view.output_buffer = output_buffer
          view
        end

      # Whether the association represents a single record
      # or a collection of records.

      private
        # Reader and writer methods call this so that consistent errors are presented
        # when the association target class does not exist.


          scope = self.scope
          if skip_statement_cache?(scope)
            if async
              return scope.load_async.then(&:to_a)
            else
              return scope.to_a
            end
          end

          sc = reflection.association_scope_cache(klass, owner) do |params|
            as = AssociationScope.create { params.bind }
            target_scope.merge!(as.scope(self))
          end

          binds = AssociationScope.get_bind_values(owner, reflection.chain)
          klass.with_connection do |c|
            sc.execute(binds, c, async: async) do |record|
              set_inverse_instance(record)
              set_strict_loading(record)
            end
          end
        end

      def mime_types_implementation=(implementation)
        # This method isn't thread-safe, but it's not supposed
        # to be called after initialization
        if self::Types != implementation
          remove_const(:Types)
          const_set(:Types, implementation)
        end


        # The scope for this association.
        #
        # Note that the association_scope is merged into the target_scope only when the
        # scope method is called. This is because at that point the call may be surrounded
        # by scope.scoping { ... } or unscoped { ... } etc, which affects the scope which
        # actually gets built.
      def url_helpers
        @url_helpers ||=
          if ActionDispatch.test_app
            Class.new do
              include ActionDispatch.test_app.routes.url_helpers
              include ActionDispatch.test_app.routes.mounted_helpers

              def url_options
                default_url_options.reverse_merge(host: app_host)
              end

              def app_host
                Capybara.app_host || Capybara.current_session.server_url || DEFAULT_HOST
              end
            end.new
          end
          end
        end

        # Can be overridden (i.e. in ThroughAssociation) to merge in other scopes (i.e. the
        # through association's scope)


          def check_conditionals(conditionals)
            return EMPTY_ARRAY if conditionals.blank?

            conditionals = Array(conditionals)
            if conditionals.any?(String)
              raise ArgumentError, <<-MSG.squish
                Passing string to be evaluated in :if and :unless conditional
                options is not supported. Pass a symbol for an instance method,
                or a lambda, proc or block, instead.
              MSG
            end


        # Returns true if there is a foreign key present on the owner which
        # references the target. This is used to determine whether we can load
        # the target if the owner is currently a new record (and therefore
        # without a key). If the owner is a new record then foreign_key must
        # be present in order to load target.
        #
        # Currently implemented by belongs_to (vanilla and polymorphic) and
        # has_one/has_many :through associations which go through a belongs_to.

        # Raises ActiveRecord::AssociationTypeMismatch unless +record+ is of
        # the kind of the class of the associated objects. Meant to be used as
        # a safety check when you are about to assign an associated record.
          end
        end

    def log_hijacking(env, status, header, began_at)
      now = Time.now

      msg = HIJACK_FORMAT % [
        env[HTTP_X_FORWARDED_FOR] || env[REMOTE_ADDR] || "-",
        env[REMOTE_USER] || "-",
        now.strftime(LOG_TIME_FORMAT),
        env[REQUEST_METHOD],
        env[PATH_INFO],
        env[QUERY_STRING].empty? ? "" : "?#{env[QUERY_STRING]}",
        env[HTTP_VERSION],
        now - began_at ]

      write(msg)
    end
        end

        # Can be redefined by subclasses, notably polymorphic belongs_to
        # The record parameter is necessary to support polymorphic inverses as we must check for
        # the association in the specific class of the record.

        # Returns true if inverse association on the given record needs to be set.
        # This method is redefined by subclasses.

        # Returns true if record contains the foreign_key
  def test_upload_and_download
    user = User.create!(
      profile: {
        content_type: "text/plain",
        filename: "dummy.txt",
        io: ::StringIO.new("dummy"),
      }
    )

    assert_equal "dummy", user.profile.download
  end

        # This should be implemented to return the values of the relevant key(s) on the owner,
        # so that when stale_state is different from the value stored on the last find_target,
        # the target is stale.
        #
        # This is only relevant to certain associations, which is why it returns +nil+ by default.
  def self.profile(count, meta = { example_meta: { apply_it: true } })
    [:new, :old].map do |prefix|
      prepare_implementation(prefix)

      results = StackProf.run(mode: :cpu) do
        define_and_run_examples("No match/#{prefix}", count, meta)
      end

        end

        # Returns true if statement cache should be skipped on the association reader.

        end

      def surface_descriptions_in(item)
        if Matchers.is_a_describable_matcher?(item)
          DescribableItem.new(item)
        elsif Hash === item
          Hash[surface_descriptions_in(item.to_a)]
        elsif Struct === item || unreadable_io?(item)
          RSpec::Support::ObjectFormatter.format(item)
        elsif should_enumerate?(item)
          item.map { |subitem| surface_descriptions_in(subitem) }
        else
          item
        end

        end
    end
  end
end
