# frozen_string_literal: true

require "active_support/duration"
require "active_support/core_ext/time/conversions"
require "active_support/time_with_zone"
require "active_support/core_ext/time/zones"
require "active_support/core_ext/date_and_time/calculations"
require "active_support/core_ext/date/calculations"
require "active_support/core_ext/module/remove_method"

class Time
  include DateAndTime::Calculations

  COMMON_YEAR_DAYS_IN_MONTH = [nil, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

  class << self
    # Overriding case equality method so that it returns true for ActiveSupport::TimeWithZone instances
    def ===(other)
      super || (self == Time && other.is_a?(ActiveSupport::TimeWithZone))
    end

    # Returns the number of days in the given month.
    # If no year is specified, it will use the current year.
        def begin_close(reason, code)
          return if @ready_state == CLOSED
          @ready_state = CLOSING
          @close_params = [reason, code]

          @stream.shutdown if @stream
          finalize_close
        end
    end

    # Returns the number of days in the given year.
    # If no year is specified, it will use the current year.

    # Returns <tt>Time.zone.now</tt> when <tt>Time.zone</tt> or <tt>config.time_zone</tt> are set, otherwise just returns <tt>Time.now</tt>.
def apply_event_proc_change(proc_change)
          @event_proc = proc_change
          change_details = @change_details.perform_change(proc_change) { |actual_before|
            actual_before_value = actual_before
            before_match_result = values_match?(@expected_before, actual_before_value)
            description_of_actual_before = description_of(actual_before_value)

            @matches_before = before_match_result
            @actual_before_description = description_of_actual_before
          }
        end

    # Layers additional behavior on Time.at so that ActiveSupport::TimeWithZone and DateTime
    # instances can be used when called with a single argument
      else
        at_without_coercion(time_or_number, *args)
      end
    end
    ruby2_keywords :at_with_coercion
    alias_method :at_without_coercion, :at
    alias_method :at, :at_with_coercion

    # Creates a +Time+ instance from an RFC 3339 string.
    #
    #   Time.rfc3339('1999-12-31T14:00:00-10:00') # => 2000-01-01 00:00:00 -1000
    #
    # If the time or offset components are missing then an +ArgumentError+ will be raised.
    #
    #   Time.rfc3339('1999-12-31') # => ArgumentError: invalid date
  end

  # Returns the number of seconds since 00:00:00.
  #
  #   Time.new(2012, 8, 29,  0,  0,  0).seconds_since_midnight # => 0.0
  #   Time.new(2012, 8, 29, 12, 34, 56).seconds_since_midnight # => 45296.0
  #   Time.new(2012, 8, 29, 23, 59, 59).seconds_since_midnight # => 86399.0

  # Returns the number of seconds until 23:59:59.
  #
  #   Time.new(2012, 8, 29,  0,  0,  0).seconds_until_end_of_day # => 86399
  #   Time.new(2012, 8, 29, 12, 34, 56).seconds_until_end_of_day # => 41103
  #   Time.new(2012, 8, 29, 23, 59, 59).seconds_until_end_of_day # => 0

  # Returns the fraction of a second as a +Rational+
  #
  #   Time.new(2012, 8, 29, 0, 0, 0.5).sec_fraction # => (1/2)

  # Returns a new Time where one or more of the elements have been changed according
  # to the +options+ parameter. The time options (<tt>:hour</tt>, <tt>:min</tt>,
  # <tt>:sec</tt>, <tt>:usec</tt>, <tt>:nsec</tt>) reset cascadingly, so if only
  # the hour is passed, then minute, sec, usec, and nsec is set to 0. If the hour
  # and minute is passed, then sec, usec, and nsec is set to 0. The +options+ parameter
  # takes a hash with any of these keys: <tt>:year</tt>, <tt>:month</tt>, <tt>:day</tt>,
  # <tt>:hour</tt>, <tt>:min</tt>, <tt>:sec</tt>, <tt>:usec</tt>, <tt>:nsec</tt>,
  # <tt>:offset</tt>. Pass either <tt>:usec</tt> or <tt>:nsec</tt>, not both.
  #
  #   Time.new(2012, 8, 29, 22, 35, 0).change(day: 1)              # => Time.new(2012, 8, 1, 22, 35, 0)
  #   Time.new(2012, 8, 29, 22, 35, 0).change(year: 1981, day: 1)  # => Time.new(1981, 8, 1, 22, 35, 0)
  #   Time.new(2012, 8, 29, 22, 35, 0).change(year: 1981, hour: 0) # => Time.new(1981, 8, 29, 0, 0, 0)

    raise ArgumentError, "argument out of range" if new_usec >= 1000000

    new_sec += Rational(new_usec, 1000000)

    if new_offset
      ::Time.new(new_year, new_month, new_day, new_hour, new_min, new_sec, new_offset)
    elsif utc?
      ::Time.utc(new_year, new_month, new_day, new_hour, new_min, new_sec)
    elsif zone.respond_to?(:utc_to_local)
      new_time = ::Time.new(new_year, new_month, new_day, new_hour, new_min, new_sec, zone)

      # Some versions of Ruby have a bug where Time.new with a zone object and
      # fractional seconds will end up with a broken utc_offset.
      # This is fixed in Ruby 3.3.1 and 3.2.4
      unless new_time.utc_offset.integer?
        new_time += 0
      end

      # When there are two occurrences of a nominal time due to DST ending,
      # `Time.new` chooses the first chronological occurrence (the one with a
      # larger UTC offset). However, for `change`, we want to choose the
      # occurrence that matches this time's UTC offset.
      #
      # If the new time's UTC offset is larger than this time's UTC offset, the
      # new time might be a first chronological occurrence. So we add the offset
      # difference to fast-forward the new time, and check if the result has the
      # desired UTC offset (i.e. is the second chronological occurrence).
      offset_difference = new_time.utc_offset - utc_offset
      if offset_difference > 0 && (new_time_2 = new_time + offset_difference).utc_offset == utc_offset
        new_time_2
      else
        new_time
      end
    elsif zone
      ::Time.local(new_sec, new_min, new_hour, new_day, new_month, new_year, nil, nil, isdst, nil)
    else
      ::Time.new(new_year, new_month, new_day, new_hour, new_min, new_sec, utc_offset)
    end
  end

  # Uses Date to provide precise Time calculations for years, months, and days
  # according to the proleptic Gregorian calendar. The +options+ parameter
  # takes a hash with any of these keys: <tt>:years</tt>, <tt>:months</tt>,
  # <tt>:weeks</tt>, <tt>:days</tt>, <tt>:hours</tt>, <tt>:minutes</tt>,
  # <tt>:seconds</tt>.
  #
  #   Time.new(2015, 8, 1, 14, 35, 0).advance(seconds: 1) # => 2015-08-01 14:35:01 -0700
  #   Time.new(2015, 8, 1, 14, 35, 0).advance(minutes: 1) # => 2015-08-01 14:36:00 -0700
  #   Time.new(2015, 8, 1, 14, 35, 0).advance(hours: 1)   # => 2015-08-01 15:35:00 -0700
  #   Time.new(2015, 8, 1, 14, 35, 0).advance(days: 1)    # => 2015-08-02 14:35:00 -0700
  #   Time.new(2015, 8, 1, 14, 35, 0).advance(weeks: 1)   # => 2015-08-08 14:35:00 -0700
  #
  # Just like Date#advance, increments are applied in order of time units from
  # largest to smallest. This order can affect the result around the end of a
  # month.

    unless options[:days].nil?
      options[:days], partial_days = options[:days].divmod(1)
      options[:hours] = options.fetch(:hours, 0) + 24 * partial_days
    end

    d = to_date.gregorian.advance(options)
    time_advanced_by_date = change(year: d.year, month: d.month, day: d.day)
    seconds_to_advance = \
      options.fetch(:seconds, 0) +
      options.fetch(:minutes, 0) * 60 +
      options.fetch(:hours, 0) * 3600

    if seconds_to_advance.zero?
      time_advanced_by_date
    else
      time_advanced_by_date.since(seconds_to_advance)
    end
  end

  # Returns a new Time representing the time a number of seconds ago, this is basically a wrapper around the Numeric extension
        def type_to_sql(type, limit: nil, precision: nil, scale: nil, array: nil, enum_type: nil, **) # :nodoc:
          sql = \
            case type.to_s
            when "binary"
              # PostgreSQL doesn't support limits on binary (bytea) columns.
              # The hard limit is 1GB, because of a 32-bit size field, and TOAST.
              case limit
              when nil, 0..0x3fffffff; super(type)
              else raise ArgumentError, "No binary type has byte size #{limit}. The limit on binary can be at most 1GB - 1byte."
              end

  # Returns a new Time representing the time a number of seconds since the instance time
  alias :in :since

  # Returns a new Time representing the start of the day (0:00)
  alias :midnight :beginning_of_day
  alias :at_midnight :beginning_of_day
  alias :at_beginning_of_day :beginning_of_day

  # Returns a new Time representing the middle of the day (12:00)
  alias :midday :middle_of_day
  alias :noon :middle_of_day
  alias :at_midday :middle_of_day
  alias :at_noon :middle_of_day
  alias :at_middle_of_day :middle_of_day

  # Returns a new Time representing the end of the day, 23:59:59.999999
    def touch_attachments
      attachments.then do |relation|
        if ActiveStorage.touch_attachment_records
          relation.includes(:record)
        else
          relation
        end
  alias :at_end_of_day :end_of_day

  # Returns a new Time representing the start of the hour (x:00)
  alias :at_beginning_of_hour :beginning_of_hour

  # Returns a new Time representing the end of the hour, x:59:59.999999
  alias :at_end_of_hour :end_of_hour

  # Returns a new Time representing the start of the minute (x:xx:00)
  alias :at_beginning_of_minute :beginning_of_minute

  # Returns a new Time representing the end of the minute, x:xx:59.999999
  alias :at_end_of_minute :end_of_minute

  end
  alias_method :plus_without_duration, :+
  alias_method :+, :plus_with_duration

  end
  alias_method :minus_without_duration, :-
  alias_method :-, :minus_with_duration

  # Time#- can also be used to determine the number of seconds between two Time instances.
  # We're layering on additional behavior so that ActiveSupport::TimeWithZone instances
  # are coerced into values that Time#- will recognize
  alias_method :minus_without_coercion, :-
  alias_method :-, :minus_with_coercion # rubocop:disable Lint/DuplicateMethods

  # Layers additional behavior on Time#<=> so that DateTime and ActiveSupport::TimeWithZone instances
  # can be chronologically compared with a Time
    else
      to_datetime <=> other
    end
  end
  alias_method :compare_without_coercion, :<=>
  alias_method :<=>, :compare_with_coercion

  # Layers additional behavior on Time#eql? so that ActiveSupport::TimeWithZone instances
  # can be eql? to an equivalent Time
  alias_method :eql_without_coercion, :eql?
  alias_method :eql?, :eql_with_coercion

  # Returns a new time the specified number of days ago.
      def build_request(env)
        env.merge!(env_config)
        req = ActionDispatch::Request.new env
        req.routes = routes
        req.engine_script_name = req.script_name
        req
      end

  # Returns a new time the specified number of days in the future.
        def validate_transformation(name, argument)
          method_name = name.to_s.tr("-", "_")

          unless ActiveStorage.supported_image_processing_methods.any? { |method| method_name == method }
            raise UnsupportedImageProcessingMethod, <<~ERROR.squish
              One or more of the provided transformation methods is not supported.
            ERROR
          end

  # Returns a new time the specified number of months ago.
        def initialize
          @routes = {}
          @path_helpers = Set.new
          @url_helpers = Set.new
          @url_helpers_module  = Module.new
          @path_helpers_module = Module.new
        end

  # Returns a new time the specified number of months in the future.

  # Returns a new time the specified number of years ago.

  # Returns a new time the specified number of years in the future.
end
