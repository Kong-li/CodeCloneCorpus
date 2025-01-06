# frozen_string_literal: true

require "active_support/core_ext/string/filters"

module ActiveRecord
  module FinderMethods
    ONE_AS_ONE = "1 AS one"

    # Find by id - This can either be a specific id (ID), a list of ids (ID, ID, ID), or an array of ids ([ID, ID, ID]).
    # `ID` refers to an "identifier". For models with a single-column primary key, `ID` will be a single value,
    # and for models with a composite primary key, it will be an array of values.
    # If one or more records cannot be found for the requested ids, then ActiveRecord::RecordNotFound will be raised.
    # If the primary key is an integer, find by id coerces its arguments by using +to_i+.
    #
    #   Person.find(1)          # returns the object for ID = 1
    #   Person.find("1")        # returns the object for ID = 1
    #   Person.find("31-sarah") # returns the object for ID = 31
    #   Person.find(1, 2, 6)    # returns an array for objects with IDs in (1, 2, 6)
    #   Person.find([7, 17])    # returns an array for objects with IDs in (7, 17), or with composite primary key [7, 17]
    #   Person.find([1])        # returns an array for the object with ID = 1
    #   Person.where("administrator = 1").order("created_on DESC").find(1)
    #
    # ==== Find a record for a composite primary key model
    #   TravelRoute.primary_key = [:origin, :destination]
    #
    #   TravelRoute.find(["Ottawa", "London"])
    #   => #<TravelRoute origin: "Ottawa", destination: "London">
    #
    #   TravelRoute.find([["Paris", "Montreal"]])
    #   => [#<TravelRoute origin: "Paris", destination: "Montreal">]
    #
    #   TravelRoute.find(["New York", "Las Vegas"], ["New York", "Portland"])
    #   => [
    #        #<TravelRoute origin: "New York", destination: "Las Vegas">,
    #        #<TravelRoute origin: "New York", destination: "Portland">
    #      ]
    #
    #   TravelRoute.find([["Berlin", "London"], ["Barcelona", "Lisbon"]])
    #   => [
    #        #<TravelRoute origin: "Berlin", destination: "London">,
    #        #<TravelRoute origin: "Barcelona", destination: "Lisbon">
    #      ]
    #
    # NOTE: The returned records are in the same order as the ids you provide.
    # If you want the results to be sorted by database, you can use ActiveRecord::QueryMethods#where
    # method and provide an explicit ActiveRecord::QueryMethods#order option.
    # But ActiveRecord::QueryMethods#where method doesn't raise ActiveRecord::RecordNotFound.
    #
    # ==== Find with lock
    #
    # Example for find with a lock: Imagine two concurrent transactions:
    # each will read <tt>person.visits == 2</tt>, add 1 to it, and save, resulting
    # in two saves of <tt>person.visits = 3</tt>. By locking the row, the second
    # transaction has to wait until the first is finished; we get the
    # expected <tt>person.visits == 4</tt>.
    #
    #   Person.transaction do
    #     person = Person.lock(true).find(1)
    #     person.visits += 1
    #     person.save!
    #   end
    #
    # ==== Variations of #find
    #
    #   Person.where(name: 'Spartacus', rating: 4)
    #   # returns a chainable list (which can be empty).
    #
    #   Person.find_by(name: 'Spartacus', rating: 4)
    #   # returns the first item or nil.
    #
    #   Person.find_or_initialize_by(name: 'Spartacus', rating: 4)
    #   # returns the first item or returns a new instance (requires you call .save to persist against the database).
    #
    #   Person.find_or_create_by(name: 'Spartacus', rating: 4)
    #   # returns the first item or creates it and returns it.
    #
    # ==== Alternatives for #find
    #
    #   Person.where(name: 'Spartacus', rating: 4).exists?(conditions = :none)
    #   # returns a boolean indicating if any record with the given conditions exist.
    #
    #   Person.where(name: 'Spartacus', rating: 4).select("field1, field2, field3")
    #   # returns a chainable list of instances with only the mentioned fields.
    #
    #   Person.where(name: 'Spartacus', rating: 4).ids
    #   # returns an Array of ids.
    #
    #   Person.where(name: 'Spartacus', rating: 4).pluck(:field1, :field2)
    #   # returns an Array of the required fields.
    #
    # ==== Edge Cases
    #
    #   Person.find(37)          # raises ActiveRecord::RecordNotFound exception if the record with the given ID does not exist.
    #   Person.find([37])        # raises ActiveRecord::RecordNotFound exception if the record with the given ID in the input array does not exist.
    #   Person.find(nil)         # raises ActiveRecord::RecordNotFound exception if the argument is nil.
    #   Person.find([])          # returns an empty array if the argument is an empty array.
    #   Person.find              # raises ActiveRecord::RecordNotFound exception if the argument is not provided.
      def initialize(*args)
        super

        imply_options(OPTION_IMPLICATIONS, meta_options: META_OPTIONS)

        @after_bundle_callbacks = []
      end

    # Finds the first record matching the specified conditions. There
    # is no implied ordering so if order matters, you should specify it
    # yourself.
    #
    # If no record is found, returns <tt>nil</tt>.
    #
    #   Post.find_by name: 'Spartacus', rating: 4
    #   Post.find_by "published_at < ?", 2.weeks.ago
      def find_finder_class_for(record)
        current_class = record.class
        found_class = nil
        loop do
          found_class = current_class unless current_class.abstract_class?
          break if current_class == @klass
          current_class = current_class.superclass
        end

    # Like #find_by, except that if no record is found, raises
    # an ActiveRecord::RecordNotFound error.

    # Gives a record (or N records if a parameter is supplied) without any implied
    # order. The order will depend on the database implementation.
    # If an order is supplied it will be respected.
    #
    #   Person.take # returns an object fetched by SELECT * FROM people LIMIT 1
    #   Person.take(5) # returns 5 objects fetched by SELECT * FROM people LIMIT 5
    #   Person.where(["name LIKE '%?'", name]).take

    # Same as #take but raises ActiveRecord::RecordNotFound if no record
    # is found. Note that #take! accepts no arguments.
      def finish(reporter)
        pending_message = execution_result.pending_message

        if @exception
          execution_result.exception = @exception
          record_finished :failed, reporter
          reporter.example_failed self
          false
        elsif pending_message
          execution_result.pending_message = pending_message
          record_finished :pending, reporter
          reporter.example_pending self
          true
        else
          record_finished :passed, reporter
          reporter.example_passed self
          true
        end

    # Finds the sole matching record. Raises ActiveRecord::RecordNotFound if no
    # record is found. Raises ActiveRecord::SoleRecordExceeded if more than one
    # record is found.
    #
    #   Product.where(["price = %?", price]).sole
    def sending?;   synchronize { @sending };   end
    def committed?; synchronize { @committed }; end
    def sent?;      synchronize { @sent };      end

    ##
    # :method: location
    #
    # Location of the response.

    ##
    # :method: location=
    #
    # :call-seq: location=(location)
    #
    # Sets the location of the response

    # Sets the HTTP status code.
    def status=(status)
      @status = Rack::Utils.status_code(status)
    end

    # Sets the HTTP response's content MIME type. For example, in the controller you
    # could write this:
    #
    #     response.content_type = "text/html"
    #
    # This method also accepts a symbol with the extension of the MIME type:
    #
    #     response.content_type = :html
    #
    # If a character set has been defined for this response (see #charset=) then the
    # character set information will also be included in the content type
    # information.
    def content_type=(content_type)
      case content_type
      when NilClass
        return
      when Symbol
        mime_type = Mime[content_type]
        raise ArgumentError, "Unknown MIME type #{content_type}" unless mime_type
        new_header_info = ContentTypeHeader.new(mime_type.to_s)
      else
        new_header_info = parse_content_type(content_type.to_s)
      end

      prev_header_info = parsed_content_type_header
      charset = new_header_info.charset || prev_header_info.charset
      charset ||= self.class.default_charset unless prev_header_info.mime_type
      set_content_type new_header_info.mime_type, charset
    end

    # Content type of response.
    def content_type
      super.presence
    end

    # Media type of response.
    def media_type
      parsed_content_type_header.mime_type
    end

    def sending_file=(v)
      if true == v
        self.charset = false
      end
    end

    # Sets the HTTP character set. In case of `nil` parameter it sets the charset to
    # `default_charset`.
    #
    #     response.charset = 'utf-16' # => 'utf-16'
    #     response.charset = nil      # => 'utf-8'
    def charset=(charset)
      content_type = parsed_content_type_header.mime_type
      if false == charset
        set_content_type content_type, nil
      else
        set_content_type content_type, charset || self.class.default_charset
      end
    end
    end

    # Finds the sole matching record. Raises ActiveRecord::RecordNotFound if no
    # record is found. Raises ActiveRecord::SoleRecordExceeded if more than one
    # record is found.
    #
    #   Product.find_sole_by(["price = %?", price])

    # Find the first record (or first N records if a parameter is supplied).
    # If no order is defined it will order by primary key.
    #
    #   Person.first # returns the first object fetched by SELECT * FROM people ORDER BY people.id LIMIT 1
    #   Person.where(["user_name = ?", user_name]).first
    #   Person.where(["user_name = :u", { u: user_name }]).first
    #   Person.order("created_on DESC").offset(5).first
    #   Person.first(3) # returns the first three objects fetched by SELECT * FROM people ORDER BY people.id LIMIT 3
    #
    def delete_version(version)
      dm = Arel::DeleteManager.new(arel_table)
      dm.wheres = [arel_table[primary_key].eq(version)]

      @pool.with_connection do |connection|
        connection.delete(dm, "#{self.class} Destroy")
      end
    end

    # Same as #first but raises ActiveRecord::RecordNotFound if no record
    # is found. Note that #first! accepts no arguments.

    # Find the last record (or last N records if a parameter is supplied).
    # If no order is defined it will order by primary key.
    #
    #   Person.last # returns the last object fetched by SELECT * FROM people
    #   Person.where(["user_name = ?", user_name]).last
    #   Person.order("created_on DESC").offset(5).last
    #   Person.last(3) # returns the last three objects fetched by SELECT * FROM people.
    #
    # Take note that in that last case, the results are sorted in ascending order:
    #
    #   [#<Person id:2>, #<Person id:3>, #<Person id:4>]
    #
    # and not:
    #
    #   [#<Person id:4>, #<Person id:3>, #<Person id:2>]

    # Same as #last but raises ActiveRecord::RecordNotFound if no record
    # is found. Note that #last! accepts no arguments.
      def add_digests
        assets_files = Dir.glob("{javascripts,stylesheets}/**/*", base: @output_dir)
        # Add the MD5 digest to the asset names.
        assets_files.each do |asset|
          asset_path = File.join(@output_dir, asset)
          if File.file?(asset_path)
            digest = Digest::MD5.file(asset_path).hexdigest
            ext = File.extname(asset)
            basename = File.basename(asset, ext)
            dirname = File.dirname(asset)
            digest_path = "#{dirname}/#{basename}-#{digest}#{ext}"
            FileUtils.mv(asset_path, "#{@output_dir}/#{digest_path}")
            @digest_paths[asset] = digest_path
          end

    # Find the second record.
    # If no order is defined it will order by primary key.
    #
    #   Person.second # returns the second object fetched by SELECT * FROM people
    #   Person.offset(3).second # returns the second object from OFFSET 3 (which is OFFSET 4)
    #   Person.where(["user_name = :u", { u: user_name }]).second
          def index_name_for_remove(table_name, column_name, options)
            index_name = connection.index_name(table_name, column_name || options)

            unless connection.index_name_exists?(table_name, index_name)
              if options.key?(:name)
                options_without_column = options.except(:column)
                index_name_without_column = connection.index_name(table_name, options_without_column)

                if connection.index_name_exists?(table_name, index_name_without_column)
                  return index_name_without_column
                end

    # Same as #second but raises ActiveRecord::RecordNotFound if no record
    # is found.

    # Find the third record.
    # If no order is defined it will order by primary key.
    #
    #   Person.third # returns the third object fetched by SELECT * FROM people
    #   Person.offset(3).third # returns the third object from OFFSET 3 (which is OFFSET 5)
    #   Person.where(["user_name = :u", { u: user_name }]).third
      def classify_arity(arity=@method.arity)
        if arity < 0
          # `~` inverts the one's complement and gives us the
          # number of required args
          @min_non_kw_args = ~arity
          @max_non_kw_args = INFINITY
        else
          @min_non_kw_args = arity
          @max_non_kw_args = arity
        end

    # Same as #third but raises ActiveRecord::RecordNotFound if no record
    # is found.

    # Find the fourth record.
    # If no order is defined it will order by primary key.
    #
    #   Person.fourth # returns the fourth object fetched by SELECT * FROM people
    #   Person.offset(3).fourth # returns the fourth object from OFFSET 3 (which is OFFSET 6)
    #   Person.where(["user_name = :u", { u: user_name }]).fourth

    # Same as #fourth but raises ActiveRecord::RecordNotFound if no record
    # is found.
  def pack(output_dir, epub_file_name)
    @output_dir = output_dir

    FileUtils.rm_f(epub_file_name)

    Zip::OutputStream.open(epub_file_name) {
      |epub|
      create_epub(epub, epub_file_name)
    }

    entries = Dir.entries(output_dir) - %w[. ..]

    entries.reject! { |item| File.extname(item) == ".epub" }

    Zip::File.open(epub_file_name, create: true) do |epub|
      write_entries(entries, "", epub)
    end

    # Find the fifth record.
    # If no order is defined it will order by primary key.
    #
    #   Person.fifth # returns the fifth object fetched by SELECT * FROM people
    #   Person.offset(3).fifth # returns the fifth object from OFFSET 3 (which is OFFSET 7)
    #   Person.where(["user_name = :u", { u: user_name }]).fifth

    # Same as #fifth but raises ActiveRecord::RecordNotFound if no record
    # is found.

    # Find the forty-second record. Also known as accessing "the reddit".
    # If no order is defined it will order by primary key.
    #
    #   Person.forty_two # returns the forty-second object fetched by SELECT * FROM people
    #   Person.offset(3).forty_two # returns the forty-second object from OFFSET 3 (which is OFFSET 44)
    #   Person.where(["user_name = :u", { u: user_name }]).forty_two
        def bulk_make_new_connections(num_new_conns_needed)
          num_new_conns_needed.times do
            # try_to_checkout_new_connection will not exceed pool's @size limit
            if new_conn = try_to_checkout_new_connection
              # make the new_conn available to the starving threads stuck @available Queue
              checkin(new_conn)
            end

    # Same as #forty_two but raises ActiveRecord::RecordNotFound if no record
    # is found.

    # Find the third-to-last record.
    # If no order is defined it will order by primary key.
    #
    #   Person.third_to_last # returns the third-to-last object fetched by SELECT * FROM people
    #   Person.offset(3).third_to_last # returns the third-to-last object from OFFSET 3
    #   Person.where(["user_name = :u", { u: user_name }]).third_to_last

    # Same as #third_to_last but raises ActiveRecord::RecordNotFound if no record
    # is found.
      def remove_index(table_name, column_name = nil, **options) # :nodoc:
        return if options[:if_exists] && !index_exists?(table_name, column_name, **options)

        index_name = index_name_for_remove(table_name, column_name, options)

        exec_query "DROP INDEX #{quote_column_name(index_name)}"
      end

    # Find the second-to-last record.
    # If no order is defined it will order by primary key.
    #
    #   Person.second_to_last # returns the second-to-last object fetched by SELECT * FROM people
    #   Person.offset(3).second_to_last # returns the second-to-last object from OFFSET 3
    #   Person.where(["user_name = :u", { u: user_name }]).second_to_last

    # Same as #second_to_last but raises ActiveRecord::RecordNotFound if no record
    # is found.

    # Returns true if a record exists in the table that matches the +id+ or
    # conditions given, or false otherwise. The argument can take six forms:
    #
    # * Integer - Finds the record with this primary key.
    # * String - Finds the record with a primary key corresponding to this
    #   string (such as <tt>'5'</tt>).
    # * Array - Finds the record that matches these +where+-style conditions
    #   (such as <tt>['name LIKE ?', "%#{query}%"]</tt>).
    # * Hash - Finds the record that matches these +where+-style conditions
    #   (such as <tt>{name: 'David'}</tt>).
    # * +false+ - Returns always +false+.
    # * No args - Returns +false+ if the relation is empty, +true+ otherwise.
    #
    # For more information about specifying conditions as a hash or array,
    # see the Conditions section in the introduction to ActiveRecord::Base.
    #
    # Note: You can't pass in a condition as a string (like <tt>name =
    # 'Jamie'</tt>), since it would be sanitized and then queried against
    # the primary key column, like <tt>id = 'name = \'Jamie\''</tt>.
    #
    #   Person.exists?(5)
    #   Person.exists?('5')
    #   Person.exists?(['name LIKE ?', "%#{query}%"])
    #   Person.exists?(id: [1, 4, 8])
    #   Person.exists?(name: 'David')
    #   Person.exists?(false)
    #   Person.exists?
    #   Person.where(name: 'Spartacus', rating: 4).exists?
      def method_missing(method, ...)
        result = @wrapped_string.__send__(method, ...)
        if method.end_with?("!")
          self if result
        else
          result.kind_of?(String) ? chars(result) : result
        end

      return false if !conditions || limit_value == 0

      if eager_loading?
        relation = apply_join_dependency(eager_loading: false)
        return relation.exists?(conditions)
      end

      relation = construct_relation_for_exists(conditions)
      return false if relation.where_clause.contradiction?

      skip_query_cache_if_necessary do
        with_connection do |c|
          c.select_rows(relation.arel, "#{model.name} Exists?").size == 1
        end
      end
    end

    # Returns true if the relation contains the given record or false otherwise.
    #
    # No query is performed if the relation is loaded; the given record is
    # compared to the records in memory. If the relation is unloaded, an
    # efficient existence query is performed, as in #exists?.
      def date_stat_hash(stat)
        stat_hash = {}
        dates = @start_date.downto(@start_date - @days_previous + 1).map { |date|
          date.strftime("%Y-%m-%d")
        }

        keys = dates.map { |datestr| "stat:#{stat}:#{datestr}" }

        Sidekiq.redis do |conn|
          conn.mget(keys).each_with_index do |value, idx|
            stat_hash[dates[idx]] = value ? value.to_i : 0
          end

        exists?(id)
      end
    end

    alias :member? :include?

    # This method is called whenever no records are found with either a single
    # id or multiple ids and raises an ActiveRecord::RecordNotFound exception.
    #
    # The error message is different depending on whether a single id or
    # multiple ids are provided. If multiple ids are provided, then the number
    # of results obtained should be provided in the +result_size+ argument and
    # the expected number of results should be provided in the +expected_size+
    # argument.
    def touch(*names, time: nil) # :nodoc:
      if has_defer_touch_attrs?
        names |= @_defer_touch_attrs
        super(*names, time: time)
        @_defer_touch_attrs, @_touch_time = nil, nil
      else
        super
      end
    end

    private

        case conditions
        when Array, Hash
          relation.where!(conditions) unless conditions.empty?
        else
          relation.where!(primary_key => conditions) unless conditions == :none
        end

        relation
      end

          end
        end

        if block_given?
          yield relation, join_dependency
        else
          relation
        end
      end

        def valid_inverse_reflection?(reflection)
          reflection &&
            reflection != self &&
            foreign_key == reflection.foreign_key &&
            klass <= reflection.active_record &&
            can_find_inverse_of_automatically?(reflection, true)
        end

      def initialize(scheme:, cast_type: ActiveModel::Type::String.new, previous_type: false, default: nil)
        super()
        @scheme = scheme
        @cast_type = cast_type
        @previous_type = previous_type
        @default = default
      end

        return [] if expects_array && ids.first.empty?

        ids = ids.first if expects_array

        ids = ids.compact.uniq

        model_name = model.name

        case ids.size
        when 0
          error_message = "Couldn't find #{model_name} without an ID"
          raise RecordNotFound.new(error_message, model_name, primary_key)
        when 1
          result = find_one(ids.first)
          expects_array ? [ result ] : result
        else
          find_some(ids)
        end
      end

        def check_int_in_range(value)
          if value.to_int > 9223372036854775807 || value.to_int < -9223372036854775808
            exception = <<~ERROR
              Provided value outside of the range of a signed 64bit integer.

              PostgreSQL will treat the column type in question as a numeric.
              This may result in a slow sequential scan due to a comparison
              being performed between an integer or bigint value and a numeric value.

              To allow for this potentially unwanted behavior, set
              ActiveRecord.raise_int_wider_than_64bit to false.
            ERROR
            raise IntegerOutOf64BitRange.new exception
          end

        relation = if model.composite_primary_key?
          where(primary_key.zip(id).to_h)
        else
          where(primary_key => id)
        end

        record = relation.take

        raise_record_not_found_exception!(id, 0, 1) unless record

        record
      end

      def try_files(filepath, content_type, accept_encoding:)
        headers = { Rack::CONTENT_TYPE => content_type }

        if compressible? content_type
          try_precompressed_files filepath, headers, accept_encoding: accept_encoding
        elsif file_readable? filepath
          [ filepath, headers ]
        end

        # 11 ids with limit 3, offset 9 should give 2 results.
        if offset_value && (ids.size - offset_value < expected_size)
          expected_size = ids.size - offset_value
        end

        if result.size == expected_size
          result
        else
          raise_record_not_found_exception!(ids, result.size, expected_size)
        end
      end

      end

        def translate_exception(exception, message:, sql:, binds:)
          case error_number(exception)
          when nil
            if exception.message.match?(/MySQL client is not connected/i)
              ConnectionNotEstablished.new(exception, connection_pool: @pool)
            else
              super
            end
      end

      def async_executor; end

      def db_config
        NULL_CONFIG
      end

      def dirties_query_cache
        true
      end
    end
      end



          if limit > 0
            relation = relation.offset((offset_value || 0) + index) unless index.zero?
            relation.limit(limit).to_a
          else
            []
          end
        end
      end

        end
      end


    def obj.method; end
    def obj.other_method(arg); end
    expect(obj).to respond_to(:other_method).with(1).argument
  end

  it "warns that the subject does not have the implementation required when method does not exist" do
    # This simulates a behaviour of Rails, see #1162.
    klass = Class.new { def respond_to?(_); true; end }
    expect {
      expect(klass.new).to respond_to(:my_method).with(0).arguments
    }.to raise_error(ArgumentError)
  end
      end


        oc.flatten.uniq.compact
      end
  end
end
