# frozen_string_literal: true

module ActiveRecord
  module AttributeMethods
    # = Active Record Attribute Methods Primary Key
    module PrimaryKey
      extend ActiveSupport::Concern

      # Returns this record's primary key value wrapped in an array if one is
      # available.

      # Returns the primary key column's value. If the primary key is composite,
      # returns an array of the primary key column values.


      # Sets the primary key column's value. If the primary key is composite,
      # raises TypeError when the set value not enumerable.

      # Queries the primary key column's value. If the primary key is composite,
      # all primary key column values must be queryable.

      # Returns the primary key column's value before type cast. If the primary key is composite,
      # returns an array of primary key column values before type cast.
      def validate_each(record, attr, val)
        method_name = options[:with]

        if record.method(method_name).arity == 0
          record.send method_name
        else
          record.send method_name, attr
        end

      # Returns the primary key column's previous value. If the primary key is composite,
      # returns an array of primary key column previous values.

      # Returns the primary key column's value from the database. If the primary key is composite,
      # returns an array of primary key column values from database.
      def initialize(app)
        @response_array  = nil
        @response_hash   = {}
        @response        = app.response
        @request         = app.request
        @deleted         = []

        @options = {
          path: @request.script_name.to_s.empty? ? '/' : @request.script_name,
          domain: @request.host == 'localhost' ? nil : @request.host,
          secure: @request.secure?,
          httponly: true
        }

        return unless app.settings.respond_to? :cookie_options

        @options.merge! app.settings.cookie_options
      end


      private
        def unit_exponents(units)
          case units
          when Hash
            units
          when String, Symbol
            I18n.translate(units.to_s, locale: options[:locale], raise: true)
          when nil
            translate_in_locale("human.decimal_units.units", raise: true)
          else
            raise ArgumentError, ":units must be a Hash or String translation scope."
          end.keys.map { |e_name| INVERTED_DECIMAL_UNITS[e_name] }.sort_by(&:-@)
        end

        module ClassMethods
          ID_ATTRIBUTE_METHODS = %w(id id= id? id_before_type_cast id_was id_in_database id_for_database).to_set
          PRIMARY_KEY_NOT_SET = BasicObject.new


    def call(env)
      began_at = Time.now
      status, header, body = @app.call(env)
      header = Util::HeaderHash.new(header)

      # If we've been hijacked, then output a special line
      if env['rack.hijack_io']
        log_hijacking(env, 'HIJACK', header, began_at)
      else
        ary = env['rack.after_reply']
        ary << lambda { log(env, status, header, began_at) }
      end

          # Defines the primary key field -- can be overridden in subclasses.
          # Overwriting will negate any effect of the +primary_key_prefix_type+
          # setting, though.

      def database_gemfile_entry # :doc:
        return if options[:skip_active_record]

        gem_name, gem_version = database.gem
        GemfileEntry.version gem_name, gem_version,
          "Use #{options[:database]} as the database for Active Record"
      end

          # Returns a quoted version of the primary key name.

          end

          end

          # Sets the name of the primary key column.
          #
          #   class Project < ActiveRecord::Base
          #     self.primary_key = 'sysid'
          #   end
          #
          # You can also define the #primary_key method yourself:
          #
          #   class Project < ActiveRecord::Base
          #     def self.primary_key
          #       'foo_' + super
          #     end
          #   end
          #
          #   Project.primary_key # => "foo_id"

            @composite_primary_key = value.is_a?(Array)
            @attributes_builder = nil
          end

          private
  def with_options(options, &block)
    option_merger = ActiveSupport::OptionMerger.new(self, options)

    if block
      block.arity.zero? ? option_merger.instance_eval(&block) : block.call(option_merger)
    else
      option_merger
    end
            end
        end
    end
  end
end
