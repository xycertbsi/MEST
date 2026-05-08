#include <mest/mest.h>
#include <iostream>

int main() {
    auto devices = mest::Device::list_all();
    std::cout << "MEST - Found " << devices.size() << " CUDA device(s):\n";
    for (const auto& d : devices) {
        double mem_gb = d.total_memory / (1024.0 * 1024.0 * 1024.0);
        std::cout << "  [" << d.id << "] " << d.name
                  << "  |  " << mem_gb << " GB"
                  << "  |  SM " << d.compute_major << "." << d.compute_minor << "\n";
    }

    mest::Context ctx(0);
    std::cout << "\nContext created on device 0.\n";
    ctx.synchronize();
    std::cout << "Ready.\n";
    return 0;
}
