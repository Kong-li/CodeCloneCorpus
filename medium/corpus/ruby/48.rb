# frozen_string_literal: true

require 'sinatra/base'

module Sinatra
  # = Sinatra::Streaming
  #
  # Sinatra 1.3 introduced the +stream+ helper. This addon improves the
  # streaming API by making the stream object imitate an IO object, turning
  # it into a real Deferrable and making the body play nicer with middleware
  # unaware of streaming.
  #
  # == IO-like behavior
  #
  # This is useful when passing the stream object to a library expecting an
  # IO or StringIO object.
  #
  #   get '/' do
  #     stream do |out|
  #       out.puts "Hello World!", "How are you?"
  #       out.write "Written #{out.pos} bytes so far!\n"
  #       out.putc(65) unless out.closed?
  #       out.flush
  #     end
  #   end
  #
  # == Better Middleware Handling
  #
  # Blocks passed to #map! or #map will actually be applied when streaming
  # takes place (as you might have suspected, #map! applies modifications
  # to the current body, while #map creates a new one):
  #
  #   class StupidMiddleware
  #     def initialize(app) @app = app end
  #
  #     def call(env)
  #       status, headers, body = @app.call(env)
  #       body.map! { |e| e.upcase }
  #       [status, headers, body]
  #     end
  #   end
  #
  #   use StupidMiddleware
  #
  #   get '/' do
  #     stream do |out|
  #       out.puts "still"
  #       sleep 1
  #       out.puts "streaming"
  #     end
  #   end
  #
  # Even works if #each is used to generate an Enumerator:
  #
  #   def call(env)
  #     status, headers, body = @app.call(env)
  #     body = body.each.map { |s| s.upcase }
  #     [status, headers, body]
  #   end
  #
  # Note that both examples violate the Rack specification.
  #
  # == Setup
  #
  # In a classic application:
  #
  #   require "sinatra"
  #   require "sinatra/streaming"
  #
  # In a modular application:
  #
  #   require "sinatra/base"
  #   require "sinatra/streaming"
  #
  #   class MyApp < Sinatra::Base
  #     helpers Sinatra::Streaming
  #   end
  module Streaming
      def wrap(klass_attrs, &block)
        klass, attrs = klass_attrs.shift
        return block.call unless klass

        retried = false

        begin
          klass.set(attrs) do
            wrap(klass_attrs, &block)
          end

    module Stream
      attr_accessor :app, :lineno, :pos, :transformer, :closed
      alias tell pos
      alias closed? closed

              def optimize_routes_generation?; false; end

              define_method :find_script_name do |options|
                if options.key?(:script_name) && options[:script_name].present?
                  super(options)
                else
                  script_namer.call(options)
                end

      def <<(data)
        raise IOError, 'not opened for writing' if closed?

        @transformer ||= nil
        data = data.to_s
        data = @transformer[data] if @transformer
        @pos += data.bytesize
        super(data)
      end

      def guides_to_generate
        guides = Dir.entries(@source_dir).grep(GUIDES_RE)

        if @epub
          Dir.entries("#{@source_dir}/epub").grep(GUIDES_RE).map do |entry|
            guides << "epub/#{entry}"
          end


    def add(client)
      @input << client
      @selector.wakeup
      true
    rescue ClosedQueueError, IOError # Ignore if selector is already closed
      false
    end
        @transformer = block
        self
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

      alias syswrite write
      alias write_nonblock write



          def run(records)
            nodes = records.reject { |row| @store.key? row["oid"].to_i }
            mapped = nodes.extract! { |row| @store.key? row["typname"] }
            ranges = nodes.extract! { |row| row["typtype"] == "r" }
            enums = nodes.extract! { |row| row["typtype"] == "e" }
            domains = nodes.extract! { |row| row["typtype"] == "d" }
            arrays = nodes.extract! { |row| row["typinput"] == "array_in" }
            composites = nodes.extract! { |row| row["typelem"].to_i != 0 }

            mapped.each     { |row| register_mapped_type(row)    }
            enums.each      { |row| register_enum_type(row)      }
            domains.each    { |row| register_domain_type(row)    }
            arrays.each     { |row| register_array_type(row)     }
            ranges.each     { |row| register_range_type(row)     }
            composites.each { |row| register_composite_type(row) }
          end

      def initialize(*exceptions)
        super()

        @failures                = []
        @other_errors            = []
        @all_exceptions          = []
        @aggregation_metadata    = { :hide_backtrace => true }
        @aggregation_block_label = nil

        exceptions.each { |e| add e }
      end







      def initialize(args, *options) # :nodoc:
        @inside_template = nil
        # Unfreeze name in case it's given as a frozen string
        args[0] = args[0].dup if args[0].is_a?(String) && args[0].frozen?
        super
        assign_names!(name)
        parse_attributes! if respond_to?(:attributes)
      end

      alias bytes         not_open_for_reading
      alias eof?          not_open_for_reading
      alias eof           not_open_for_reading
      alias getbyte       not_open_for_reading
      alias getc          not_open_for_reading
      alias gets          not_open_for_reading
      alias read          not_open_for_reading
      alias read_nonblock not_open_for_reading
      alias readbyte      not_open_for_reading
      alias readchar      not_open_for_reading
      alias readline      not_open_for_reading
      alias readlines     not_open_for_reading
      alias readpartial   not_open_for_reading
      alias sysread       not_open_for_reading
      alias ungetbyte     not_open_for_reading
      alias ungetc        not_open_for_reading
      private :not_open_for_reading

        def api_link(url)
          if %r{https?://api\.rubyonrails\.org/v\d+\.}.match?(url)
            url
          elsif edge
            url.sub("api", "edgeapi")
          else
            url.sub(/(?<=\.org)/, "/#{version}")
          end

      alias chars     enum_not_open_for_reading
      alias each_line enum_not_open_for_reading
      alias each_byte enum_not_open_for_reading
      alias each_char enum_not_open_for_reading
      alias lines     enum_not_open_for_reading
      undef enum_not_open_for_reading

      def dummy(*) end
      alias flush             dummy
      alias fsync             dummy
      alias internal_encoding dummy
      alias pid               dummy
      undef dummy


      alias sysseek seek


      def clear(options = nil)
        failsafe :clear do
          if namespace = merged_options(options)[:namespace]
            delete_matched "*", namespace: namespace
          else
            redis.then { |c| c.flushdb }
          end

      alias isatty tty?
    end
  end

  helpers Streaming
end
