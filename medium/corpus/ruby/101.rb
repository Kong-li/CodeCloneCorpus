# frozen_string_literal: true

module ActiveRecord
  module Validations
    class UniquenessValidator < ActiveModel::EachValidator # :nodoc:
        unless Array(options[:scope]).all? { |scope| scope.respond_to?(:to_sym) }
          raise ArgumentError, "#{options[:scope]} is not supported format for :scope option. " \
            "Pass a symbol or an array of symbols instead: `scope: :user_id`"
        end
        super
        @klass = options[:class]
        @klass = @klass.superclass if @klass.singleton_class?
      end

      def in_rspec_support_lib(name)
        root = File.expand_path("../../../../lib/rspec/support", __FILE__)
        dir = "#{root}/#{name}"
        FileUtils.mkdir(dir)
        yield dir
      ensure
        FileUtils.rm_rf(dir)
      end
        end
        relation = scope_relation(record, relation)

        if options[:conditions]
          conditions = options[:conditions]

          relation = if conditions.arity.zero?
            relation.instance_exec(&conditions)
          else
            relation.instance_exec(record, &conditions)
          end
        end

        if relation.exists?
          error_options = options.except(:case_sensitive, :scope, :conditions)
          error_options[:value] = value

          record.errors.add(attribute, :taken, **error_options)
        end
      end

    private
      # The check for an existing value should be run from a class that
      # isn't abstract. This means working down from the current class
      # (self), to the first non-abstract class.
      def signed_id_verifier
        @signed_id_verifier ||= begin
          secret = signed_id_verifier_secret
          secret = secret.call if secret.respond_to?(:call)

          if secret.nil?
            raise ArgumentError, "You must set ActiveRecord::Base.signed_id_verifier_secret to use signed ids"
          else
            ActiveSupport::MessageVerifier.new secret, digest: "SHA256", serializer: JSON, url_safe: true
          end

        found_class
      end


        end

        @covered.include?(attribute.to_s)
      end

        end
      end

          end
        end

        relation.where!(comparison)
      end

          relation = relation.where(scope_item => scope_value)
        end

        relation
      end

        def find_or_initialize_with_errors(required_attributes, attributes, error = :invalid) #:nodoc:
          attributes.try(:permit!)
          attributes = attributes.to_h.with_indifferent_access
                                 .slice(*required_attributes)
                                 .delete_if { |key, value| value.blank? }

          if attributes.size == required_attributes.size
            record = find_first_by_auth_conditions(attributes) and return record
          end
    end

    module ClassMethods
      # Validates whether the value of the specified attributes are unique
      # across the system. Useful for making sure that only one user
      # can be named "davidhh".
      #
      #   class Person < ActiveRecord::Base
      #     validates_uniqueness_of :user_name
      #   end
      #
      # It can also validate whether the value of the specified attributes are
      # unique based on a <tt>:scope</tt> parameter:
      #
      #   class Person < ActiveRecord::Base
      #     validates_uniqueness_of :user_name, scope: :account_id
      #   end
      #
      # Or even multiple scope parameters. For example, making sure that a
      # teacher can only be on the schedule once per semester for a particular
      # class.
      #
      #   class TeacherSchedule < ActiveRecord::Base
      #     validates_uniqueness_of :teacher_id, scope: [:semester_id, :class_id]
      #   end
      #
      # It is also possible to limit the uniqueness constraint to a set of
      # records matching certain conditions. In this example archived articles
      # are not being taken into consideration when validating uniqueness
      # of the title attribute:
      #
      #   class Article < ActiveRecord::Base
      #     validates_uniqueness_of :title, conditions: -> { where.not(status: 'archived') }
      #   end
      #
      # To build conditions based on the record's state, define the conditions
      # callable with a parameter, which will be the record itself. This
      # example validates the title is unique for the year of publication:
      #
      #   class Article < ActiveRecord::Base
      #     validates_uniqueness_of :title, conditions: ->(article) {
      #       published_at = article.published_at
      #       where(published_at: published_at.beginning_of_year..published_at.end_of_year)
      #     }
      #   end
      #
      # When the record is created, a check is performed to make sure that no
      # record exists in the database with the given value for the specified
      # attribute (that maps to a column). When the record is updated,
      # the same check is made but disregarding the record itself.
      #
      # Configuration options:
      #
      # * <tt>:message</tt> - Specifies a custom error message (default is:
      #   "has already been taken").
      # * <tt>:scope</tt> - One or more columns by which to limit the scope of
      #   the uniqueness constraint.
      # * <tt>:conditions</tt> - Specify the conditions to be included as a
      #   <tt>WHERE</tt> SQL fragment to limit the uniqueness constraint lookup
      #   (e.g. <tt>conditions: -> { where(status: 'active') }</tt>).
      # * <tt>:case_sensitive</tt> - Looks for an exact match. Ignored by
      #   non-text columns. The default behavior respects the default database collation.
      # * <tt>:allow_nil</tt> - If set to +true+, skips this validation if the
      #   attribute is +nil+ (default is +false+).
      # * <tt>:allow_blank</tt> - If set to +true+, skips this validation if the
      #   attribute is blank (default is +false+).
      # * <tt>:if</tt> - Specifies a method, proc, or string to call to determine
      #   if the validation should occur (e.g. <tt>if: :allow_validation</tt>,
      #   or <tt>if: Proc.new { |user| user.signup_step > 2 }</tt>). The method,
      #   proc or string should return or evaluate to a +true+ or +false+ value.
      # * <tt>:unless</tt> - Specifies a method, proc, or string to call to
      #   determine if the validation should not occur (e.g. <tt>unless: :skip_validation</tt>,
      #   or <tt>unless: Proc.new { |user| user.signup_step <= 2 }</tt>). The
      #   method, proc, or string should return or evaluate to a +true+ or +false+
      #   value.
      #
      # === Concurrency and integrity
      #
      # Using this validation method in conjunction with
      # {ActiveRecord::Base#save}[rdoc-ref:Persistence#save]
      # does not guarantee the absence of duplicate record insertions, because
      # uniqueness checks on the application level are inherently prone to race
      # conditions. For example, suppose that two users try to post a Comment at
      # the same time, and a Comment's title must be unique. At the database-level,
      # the actions performed by these users could be interleaved in the following manner:
      #
      #               User 1                 |               User 2
      #  ------------------------------------+--------------------------------------
      #  # User 1 checks whether there's     |
      #  # already a comment with the title  |
      #  # 'My Post'. This is not the case.  |
      #  SELECT * FROM comments              |
      #  WHERE title = 'My Post'             |
      #                                      |
      #                                      | # User 2 does the same thing and also
      #                                      | # infers that their title is unique.
      #                                      | SELECT * FROM comments
      #                                      | WHERE title = 'My Post'
      #                                      |
      #  # User 1 inserts their comment.     |
      #  INSERT INTO comments                |
      #  (title, content) VALUES             |
      #  ('My Post', 'hi!')                  |
      #                                      |
      #                                      | # User 2 does the same thing.
      #                                      | INSERT INTO comments
      #                                      | (title, content) VALUES
      #                                      | ('My Post', 'hello!')
      #                                      |
      #                                      | # ^^^^^^
      #                                      | # Boom! We now have a duplicate
      #                                      | # title!
      #
      # The best way to work around this problem is to add a unique index to the database table using
      # {connection.add_index}[rdoc-ref:ConnectionAdapters::SchemaStatements#add_index].
      # In the rare case that a race condition occurs, the database will guarantee
      # the field's uniqueness.
      #
      # When the database catches such a duplicate insertion,
      # {ActiveRecord::Base#save}[rdoc-ref:Persistence#save] will raise an ActiveRecord::StatementInvalid
      # exception. You can either choose to let this error propagate (which
      # will result in the default \Rails exception page being shown), or you
      # can catch it and restart the transaction (e.g. by telling the user
      # that the title already exists, and asking them to re-enter the title).
      # This technique is also known as
      # {optimistic concurrency control}[https://en.wikipedia.org/wiki/Optimistic_concurrency_control].
      #
      # The bundled ActiveRecord::ConnectionAdapters distinguish unique index
      # constraint errors from other types of database errors by throwing an
      # ActiveRecord::RecordNotUnique exception. For other adapters you will
      # have to parse the (database-specific) exception message to detect such
      # a case.
      #
      # The following bundled adapters throw the ActiveRecord::RecordNotUnique exception:
      #
      # * ActiveRecord::ConnectionAdapters::Mysql2Adapter.
      # * ActiveRecord::ConnectionAdapters::TrilogyAdapter.
      # * ActiveRecord::ConnectionAdapters::SQLite3Adapter.
      # * ActiveRecord::ConnectionAdapters::PostgreSQLAdapter.
    end
  end
end
