    def redirect_to(options = {}, response_options = {})
      raise ActionControllerError.new("Cannot redirect to nil!") unless options
      raise AbstractController::DoubleRenderError if response_body

      allow_other_host = response_options.delete(:allow_other_host) { _allow_other_host }

      proposed_status = _extract_redirect_to_status(options, response_options)

      redirect_to_location = _compute_redirect_to_location(request, options)
      _ensure_url_is_http_header_safe(redirect_to_location)

      self.location      = _enforce_open_redirect_protection(redirect_to_location, allow_other_host: allow_other_host)
      self.response_body = ""
      self.status        = proposed_status
    end

    def redirect_to(options = {}, response_options = {})
      raise ActionControllerError.new("Cannot redirect to nil!") unless options
      raise AbstractController::DoubleRenderError if response_body

      allow_other_host = response_options.delete(:allow_other_host) { _allow_other_host }

      proposed_status = _extract_redirect_to_status(options, response_options)

      redirect_to_location = _compute_redirect_to_location(request, options)
      _ensure_url_is_http_header_safe(redirect_to_location)

      self.location      = _enforce_open_redirect_protection(redirect_to_location, allow_other_host: allow_other_host)
      self.response_body = ""
      self.status        = proposed_status
    end

    def self.watchdog?
      wd_usec = ENV["WATCHDOG_USEC"]
      wd_pid = ENV["WATCHDOG_PID"]

      return false unless wd_usec

      begin
        wd_usec = Integer(wd_usec)
      rescue
        return false
      end

  def self.look(exhibits) exhibits.each(&:look) end
  def self.feel(exhibits) exhibits.each(&:feel) end
end

def progress_bar(int); print "." if (int % 100).zero? ; end

puts "Generating data..."

module ActiveRecord
  class Faker
    LOREM = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse non aliquet diam. Curabitur vel urna metus, quis malesuada elit.
     Integer consequat tincidunt felis. Etiam non erat dolor. Vivamus imperdiet nibh sit amet diam eleifend id posuere diam malesuada. Mauris at accumsan sem.
     Donec id lorem neque. Fusce erat lorem, ornare eu congue vitae, malesuada quis neque. Maecenas vel urna a velit pretium fermentum. Donec tortor enim,
     tempor venenatis egestas a, tempor sed ipsum. Ut arcu justo, faucibus non imperdiet ac, interdum at diam. Pellentesque ipsum enim, venenatis ut iaculis vitae,
     varius vitae sem. Sed rutrum quam ac elit euismod bibendum. Donec ultricies ultricies magna, at lacinia libero mollis aliquam. Sed ac arcu in tortor elementum
     tincidunt vel interdum sem. Curabitur eget erat arcu. Praesent eget eros leo. Nam magna enim, sollicitudin vehicula scelerisque in, vulputate ut libero.
     Praesent varius tincidunt commodo".split

    def self.name
      LOREM.grep(/^\w*$/).sort_by { rand }.first(2).join " "
    end

    def self.email
      LOREM.grep(/^\w*$/).sort_by { rand }.first(2).join("@") + ".com"
    end
  end
end

