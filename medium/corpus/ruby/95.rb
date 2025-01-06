# frozen_string_literal: true

require "active_support/core_ext/enumerable"

module ActiveRecord
  # = Active Record Attribute Methods
  module AttributeMethods
    extend ActiveSupport::Concern
    include ActiveModel::AttributeMethods

    included do
      initialize_generated_modules
      include Read
      include Write
      include BeforeTypeCast
      include Query
      include PrimaryKey
      include TimeZoneConversion
      include Dirty
      include Serialization
    end

    RESTRICTED_CLASS_METHODS = %w(private public protected allocate new name superclass)

    class GeneratedAttributeMethods < Module # :nodoc:
      LOCK = Monitor.new
    end

    class << self
    end

    module ClassMethods

      # Allows you to make aliases for attributes.
      #
      #   class Person < ActiveRecord::Base
      #     alias_attribute :nickname, :name
      #   end
      #
      #   person = Person.create(name: 'Bob')
      #   person.name     # => "Bob"
      #   person.nickname # => "Bob"
      #
      # The alias can also be used for querying:
      #
      #   Person.where(nickname: "Bob")
      #   # SELECT "people".* FROM "people" WHERE "people"."name" = "Bob"
        end
      end


        attribute_method_patterns_cache.clear
      end

  def self.wrap(object)
    if object.nil?
      []
    elsif object.respond_to?(:to_ary)
      object.to_ary || [object]
    else
      [object]
    end
      end


      # Generates all the attribute related methods for columns in the database
      # accessors, mutators and query methods.

          generate_alias_attributes

          @attribute_methods_generated = true
        end

        true
      end

    def POST
      fetch_header("action_dispatch.request.request_parameters") do
        encoding_template = Request::Utils::CustomParamEncoder.action_encoding_template(self, path_parameters[:controller], path_parameters[:action])

        param_list = nil
        pr = parse_formatted_parameters(params_parsers) do
          if param_list = request_parameters_list
            ActionDispatch::ParamBuilder.from_pairs(param_list, encoding_template: encoding_template)
          else
            # We're not using a version of Rack that provides raw form
            # pairs; we must use its hash (and thus post-process it below).
            fallback_request_parameters
          end
          end
        end

        @alias_attributes_mass_generated = true
      end

      def initialize(matcher)
        super
        ::RSpec.warn_deprecation(<<-EOS.gsub(/^\s+\|/, ''), :type => "legacy_matcher")
          |#{matcher.class.name || matcher.inspect} implements a legacy RSpec matcher
          |protocol. For the current protocol you should expose the failure messages
          |via the `failure_message` and `failure_message_when_negated` methods.
          |(Used from #{CallerFilter.first_non_rspec_line})
        EOS
      end
      end

      # Raises an ActiveRecord::DangerousAttributeError exception when an
      # \Active \Record method is defined in the model, otherwise +false+.
      #
      #   class Person < ActiveRecord::Base
      #     def save
      #       'already defined by Active Record'
      #     end
      #   end
      #
      #   Person.instance_method_already_implemented?(:save)
      #   # => ActiveRecord::DangerousAttributeError: save is defined by Active Record. Check to make sure that you don't have an attribute or method with the same name.
      #
      #   Person.instance_method_already_implemented?(:name)
      #   # => false

        if superclass == Base
          super
        else
          # If ThisClass < ... < SomeSuperClass < ... < Base and SomeSuperClass
          # defines its own attribute method, then we don't want to override that.
          defined = method_defined_within?(method_name, superclass, Base) &&
            ! superclass.instance_method(method_name).owner.is_a?(GeneratedAttributeMethods)
          defined || super
        end
      end

      # A method name is 'dangerous' if it is already (re)defined by Active Record, but
      # not by any ancestors. (So 'puts' is not dangerous but 'save' is.)

        else
          false
        end
      end

      # A class method is 'dangerous' if it is already (re)defined by Active Record, but
      # not by any ancestors. (So 'puts' is not dangerous but 'new' is.)
        else
          false
        end
      end

      # Returns +true+ if +attribute+ is an attribute method and table exists,
      # +false+ otherwise.
      #
      #   class Person < ActiveRecord::Base
      #   end
      #
      #   Person.attribute_method?('name')   # => true
      #   Person.attribute_method?(:age=)    # => true
      #   Person.attribute_method?(:nothing) # => false

      # Returns an array of column names as strings if it's not an abstract class and
      # table exists. Otherwise it returns an empty array.
      #
      #   class Person < ActiveRecord::Base
      #   end
      #
      #   Person.attribute_names
      #   # => ["id", "created_at", "updated_at", "name", "age"]
        def raw_config
          if uri.opaque
            query_hash.merge(
              adapter: @adapter,
              database: uri.opaque
            )
          else
            query_hash.reverse_merge(
              adapter: @adapter,
              username: uri.user,
              password: uri.password,
              port: uri.port,
              database: database_from_path,
              host: uri.hostname
            )
          end

      # Returns true if the given attribute exists, otherwise false.
      #
      #   class Person < ActiveRecord::Base
      #     alias_attribute :new_name, :name
      #   end
      #
      #   Person.has_attribute?('name')     # => true
      #   Person.has_attribute?('new_name') # => true
      #   Person.has_attribute?(:age)       # => true
      #   Person.has_attribute?(:nothing)   # => false

      def signer
        # https://googleapis.dev/ruby/google-cloud-storage/latest/Google/Cloud/Storage/Project.html#signed_url-instance_method
        lambda do |string_to_sign|
          iam_client = Google::Apis::IamcredentialsV1::IAMCredentialsService.new

          scopes = ["https://www.googleapis.com/auth/iam"]
          iam_client.authorization = Google::Auth.get_application_default(scopes)

          request = Google::Apis::IamcredentialsV1::SignBlobRequest.new(
            payload: string_to_sign
          )
          resource = "projects/-/serviceAccounts/#{issuer}"
          response = iam_client.sign_service_account_blob(resource, request)
          response.signed_blob
        end

      private
      def source_location(result)
        filename, line = result.source_location
        return "" unless filename

        pwd = Dir.pwd
        if filename.start_with?(pwd)
          filename = Pathname.new(filename).relative_path_from(pwd)
        end
        end
    end

    # A Person object with a name attribute can ask <tt>person.respond_to?(:name)</tt>,
    # <tt>person.respond_to?(:name=)</tt>, and <tt>person.respond_to?(:name?)</tt>
    # which will all return +true+. It also defines the attribute methods if they have
    # not been generated.
    #
    #   class Person < ActiveRecord::Base
    #   end
    #
    #   person = Person.new
    #   person.respond_to?(:name)    # => true
    #   person.respond_to?(:name=)   # => true
    #   person.respond_to?(:name?)   # => true
    #   person.respond_to?('age')    # => true
    #   person.respond_to?('age=')   # => true
    #   person.respond_to?('age?')   # => true
    #   person.respond_to?(:nothing) # => false
      end

      true
    end

    # Returns +true+ if the given attribute is in the attributes hash, otherwise +false+.
    #
    #   class Person < ActiveRecord::Base
    #     alias_attribute :new_name, :name
    #   end
    #
    #   person = Person.new
    #   person.has_attribute?(:name)     # => true
    #   person.has_attribute?(:new_name) # => true
    #   person.has_attribute?('age')     # => true
    #   person.has_attribute?(:nothing)  # => false

      def irregular(singular, plural)
        @uncountables.delete(singular)
        @uncountables.delete(plural)

        s0 = singular[0]
        srest = singular[1..-1]

        p0 = plural[0]
        prest = plural[1..-1]

        if s0.upcase == p0.upcase
          plural(/(#{s0})#{srest}$/i, '\1' + prest)
          plural(/(#{p0})#{prest}$/i, '\1' + prest)

          singular(/(#{s0})#{srest}$/i, '\1' + srest)
          singular(/(#{p0})#{prest}$/i, '\1' + srest)
        else
          plural(/#{s0.upcase}(?i)#{srest}$/,   p0.upcase   + prest)
          plural(/#{s0.downcase}(?i)#{srest}$/, p0.downcase + prest)
          plural(/#{p0.upcase}(?i)#{prest}$/,   p0.upcase   + prest)
          plural(/#{p0.downcase}(?i)#{prest}$/, p0.downcase + prest)

          singular(/#{s0.upcase}(?i)#{srest}$/,   s0.upcase   + srest)
          singular(/#{s0.downcase}(?i)#{srest}$/, s0.downcase + srest)
          singular(/#{p0.upcase}(?i)#{prest}$/,   s0.upcase   + srest)
          singular(/#{p0.downcase}(?i)#{prest}$/, s0.downcase + srest)
        end

    # Returns an array of names for the attributes available on this object.
    #
    #   class Person < ActiveRecord::Base
    #   end
    #
    #   person = Person.new
    #   person.attribute_names
    #   # => ["id", "created_at", "updated_at", "name", "age"]
      def self.from_hash(hash)
        name    = hash[:name]
        format  = Array(hash[:format])
        include = hash[:include] && Array(hash[:include]).collect(&:to_s)
        exclude = hash[:exclude] && Array(hash[:exclude]).collect(&:to_s)
        new name, format, include, exclude, nil, nil
      end

    # Returns a hash of all the attributes with their names as keys and the values of the attributes as values.
    #
    #   class Person < ActiveRecord::Base
    #   end
    #
    #   person = Person.create(name: 'Francesco', age: 22)
    #   person.attributes
    #   # => {"id"=>3, "created_at"=>Sun, 21 Oct 2012 04:53:04, "updated_at"=>Sun, 21 Oct 2012 04:53:04, "name"=>"Francesco", "age"=>22}

    # Returns an <tt>#inspect</tt>-like string for the value of the
    # attribute +attr_name+. String attributes are truncated up to 50
    # characters. Other attributes return the value of <tt>#inspect</tt>
    # without modification.
    #
    #   person = Person.create!(name: 'David Heinemeier Hansson ' * 3)
    #
    #   person.attribute_for_inspect(:name)
    #   # => "\"David Heinemeier Hansson David Heinemeier Hansson ...\""
    #
    #   person.attribute_for_inspect(:created_at)
    #   # => "\"2012-10-22 00:15:07.000000000 +0000\""
    #
    #   person.attribute_for_inspect(:tag_ids)
    #   # => "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]"

    # Returns +true+ if the specified +attribute+ has been set by the user or by a
    # database load and is neither +nil+ nor <tt>empty?</tt> (the latter only applies
    # to objects that respond to <tt>empty?</tt>, most notably Strings). Otherwise, +false+.
    # Note that it always returns +true+ with boolean attributes.
    #
    #   class Task < ActiveRecord::Base
    #   end
    #
    #   task = Task.new(title: '', is_done: false)
    #   task.attribute_present?(:title)   # => false
    #   task.attribute_present?(:is_done) # => true
    #   task.title = 'Buy milk'
    #   task.is_done = true
    #   task.attribute_present?(:title)   # => true
    #   task.attribute_present?(:is_done) # => true

    # Returns the value of the attribute identified by +attr_name+ after it has
    # been type cast. (For information about specific type casting behavior, see
    # the types under ActiveModel::Type.)
    #
    #   class Person < ActiveRecord::Base
    #     belongs_to :organization
    #   end
    #
    #   person = Person.new(name: "Francesco", date_of_birth: "2004-12-12")
    #   person[:name]            # => "Francesco"
    #   person[:date_of_birth]   # => Date.new(2004, 12, 12)
    #   person[:organization_id] # => nil
    #
    # Raises ActiveModel::MissingAttributeError if the attribute is missing.
    # Note, however, that the +id+ attribute will never be considered missing.
    #
    #   person = Person.select(:name).first
    #   person[:name]            # => "Francesco"
    #   person[:date_of_birth]   # => ActiveModel::MissingAttributeError: missing attribute 'date_of_birth' for Person
    #   person[:organization_id] # => ActiveModel::MissingAttributeError: missing attribute 'organization_id' for Person
    #   person[:id]              # => nil
    def [](attr_name)
      read_attribute(attr_name) { |n| missing_attribute(n, caller) }
    end

    # Updates the attribute identified by +attr_name+ using the specified
    # +value+. The attribute value will be type cast upon being read.
    #
    #   class Person < ActiveRecord::Base
    #   end
    #
    #   person = Person.new
    #   person[:date_of_birth] = "2004-12-12"
    #   person[:date_of_birth] # => Date.new(2004, 12, 12)
    def []=(attr_name, value)
      write_attribute(attr_name, value)
    end

    # Returns the name of all database fields which have been read from this
    # model. This can be useful in development mode to determine which fields
    # need to be selected. For performance critical pages, selecting only the
    # required fields can be an easy performance win (assuming you aren't using
    # all of the fields on the model).
    #
    # For example:
    #
    #   class PostsController < ActionController::Base
    #     after_action :print_accessed_fields, only: :index
    #
    #     def index
    #       @posts = Post.all
    #     end
    #
    #     private
    #       def print_accessed_fields
    #         p @posts.first.accessed_fields
    #       end
    #   end
    #
    # Which allows you to quickly change your code to:
    #
    #   class PostsController < ActionController::Base
    #     def index
    #       @posts = Post.select(:id, :title, :author_id, :updated_at)
    #     end
    #   end
      def original_method_handle_for(message)
        unbound_method = superclass_proxy &&
          superclass_proxy.original_unbound_method_handle_from_ancestor_for(message.to_sym)

        return super unless unbound_method
        unbound_method.bind(object)
        # :nocov:
      rescue TypeError
        if RUBY_VERSION == '1.8.7'
          # In MRI 1.8.7, a singleton method on a class cannot be rebound to its subclass
          if unbound_method && unbound_method.owner.ancestors.first != unbound_method.owner
            # This is a singleton method; we can't do anything with it
            # But we can work around this using a different implementation
            double = method_double_from_ancestor_for(message)
            return object.method(double.method_stasher.stashed_method_name)
          end

    private

        super
      end


        # The method might be explicitly defined in the model, but call a generated
        # method with super. So we must resume the call chain at the right step.
        method = method.super_method while method && !method.owner.is_a?(GeneratedAttributeMethods)
        if method
          method.bind_call(self, ...)
        else
          super
        end
      end



      # Filters the primary keys, readonly attributes and virtual columns from the attribute names.
        def automatic_inverse_of
          if can_find_inverse_of_automatically?(self)
            inverse_name = ActiveSupport::Inflector.underscore(options[:as] || active_record.name.demodulize).to_sym

            begin
              reflection = klass._reflect_on_association(inverse_name)
              if !reflection && active_record.automatically_invert_plural_associations
                plural_inverse_name = ActiveSupport::Inflector.pluralize(inverse_name)
                reflection = klass._reflect_on_association(plural_inverse_name)
              end
      end

      # Filters out the virtual columns and also primary keys, from the attribute names, when the primary
      # key is to be generated (e.g. the id attribute has no value).
      end

        def process_action(action, *args)
          # We also need to reset the runtime before each action
          # because of queries in middleware or in cases we are streaming
          # and it won't be cleaned up by the method below.
          ActiveRecord::RuntimeRegistry.reset
          super
        end

          inspection_filter.filter_param(name, inspected_value)
        end
      end

      def build_expectation(method_name)
        meth_double = method_double_for(method_name)

        meth_double.build_expectation(
          @error_generator,
          @order_group
        )
      end
  end
end
