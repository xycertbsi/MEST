import mest.MEST;

public class Example {
    public static void main(String[] args) {
        System.out.println("MEST Java Example");
        System.out.println("=".repeat(40));

        for (String d : MEST.listDevices()) {
            System.out.println(d);
        }

        System.out.println();

        float[] a = {1f, 2f, 3f, 4f, 5f};
        float[] b = {10f, 20f, 30f, 40f, 50f};

        float[] sum = MEST.vectorAdd(a, b);
        System.out.print("vectorAdd: ");
        for (float v : sum) System.out.print(v + " ");
        System.out.println();

        float[] scaled = MEST.vectorScale(a.clone(), 3.0f);
        System.out.print("vectorScale: ");
        for (float v : scaled) System.out.print(v + " ");
        System.out.println();

        float[][] matrix = {{1f, 2f, 3f}, {4f, 5f, 6f}};
        float[][] t = MEST.matrixTranspose(matrix);
        System.out.println("matrixTranspose result rows: " + t.length + ", cols: " + t[0].length);
    }
}
